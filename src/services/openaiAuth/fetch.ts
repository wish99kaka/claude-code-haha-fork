import type { ClientOptions } from '@anthropic-ai/sdk'
import { randomUUID } from 'crypto'
import { OPENAI_CODEX_API_ENDPOINT } from './client.js'
import { ensureFreshOpenAITokens } from './index.js'
import { resolveOpenAICodexModel } from './models.js'
import { getOpenAIOAuthTokens } from './storage.js'
import { anthropicToOpenaiChat } from '../../server/proxy/transform/anthropicToOpenaiChat.js'
import { openaiChatToAnthropic } from '../../server/proxy/transform/openaiChatToAnthropic.js'
import { anthropicToOpenaiResponses } from '../../server/proxy/transform/anthropicToOpenaiResponses.js'
import { openaiChatStreamToAnthropic } from '../../server/proxy/streaming/openaiChatStreamToAnthropic.js'
import { openaiResponsesToAnthropic } from '../../server/proxy/transform/openaiResponsesToAnthropic.js'
import { openaiResponsesStreamToAnthropic } from '../../server/proxy/streaming/openaiResponsesStreamToAnthropic.js'
import type { AnthropicRequest } from '../../server/proxy/transform/types.js'
import { logForDebugging } from '../../utils/debug.js'

export const OPENAI_OAUTH_DUMMY_KEY = 'openai-oauth-dummy-key'

export function shouldUseOpenAICodexAuth(): boolean {
  const openaiTokens = getOpenAIOAuthTokens()
  return !!openaiTokens?.refreshToken
}

export function shouldUseOpenAIChatCompatibleFetch(): boolean {
  const baseUrl = process.env.ANTHROPIC_BASE_URL
  if (!baseUrl) {
    return false
  }

  try {
    const url = new URL(baseUrl)
    return url.pathname.replace(/\/+$/, '') === '/api'
  } catch {
    return false
  }
}

export function buildOpenAICodexFetch(
  fetchOverride: ClientOptions['fetch'],
  source: string | undefined,
): ClientOptions['fetch'] {
  const inner = fetchOverride ?? globalThis.fetch

  return async (input, init) => {
    const url = input instanceof Request ? new URL(input.url) : new URL(String(input))

    if (!url.pathname.endsWith('/v1/messages')) {
      return inner(input, init)
    }

    const originalBody = await readAnthropicBody(input, init)
    const mappedModel = resolveOpenAICodexModel(originalBody.model)
    const transformedBody = anthropicToOpenaiResponses({
      ...originalBody,
      model: mappedModel,
    })

    const tokens = await ensureFreshOpenAITokens()
    if (!tokens) {
      throw new Error(
        'OpenAI OAuth token is missing or expired. Run claude auth login --openai again.',
      )
    }

    const headers = new Headers()
    headers.set('Content-Type', 'application/json')
    headers.set('Authorization', `Bearer ${tokens.accessToken}`)
    if (tokens.accountId) {
      headers.set('ChatGPT-Account-Id', tokens.accountId)
    }

    logForDebugging(
      `[API REQUEST] ${url.pathname} remapped_to=OpenAI/Codex model=${mappedModel} source=${source ?? 'unknown'} request_id=${randomUUID()}`,
    )

    const upstream = await inner(OPENAI_CODEX_API_ENDPOINT, {
      method: 'POST',
      headers,
      body: JSON.stringify(transformedBody),
      signal: init?.signal,
    })

    if (!upstream.ok) {
      const errorText = await upstream.text().catch(() => '')
      return Response.json(
        {
          type: 'error',
          error: {
            type: 'api_error',
            message: `OpenAI upstream returned HTTP ${upstream.status}: ${errorText.slice(0, 500)}`,
          },
        },
        { status: upstream.status },
      )
    }

    if (transformedBody.stream) {
      if (!upstream.body) {
        return Response.json(
          {
            type: 'error',
            error: {
              type: 'api_error',
              message: 'OpenAI upstream returned no body for stream',
            },
          },
          { status: 502 },
        )
      }

      return new Response(
        openaiResponsesStreamToAnthropic(upstream.body, mappedModel),
        {
          status: 200,
          headers: {
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            Connection: 'keep-alive',
          },
        },
      )
    }

    const responseBody = await upstream.json()
    return Response.json(
      openaiResponsesToAnthropic(responseBody, mappedModel),
    )
  }
}

export function buildOpenAIChatCompatibleFetch(
  fetchOverride: ClientOptions['fetch'],
  source: string | undefined,
): ClientOptions['fetch'] {
  const inner = fetchOverride ?? globalThis.fetch

  return async (input, init) => {
    const url = input instanceof Request ? new URL(input.url) : new URL(String(input))

    if (!url.pathname.endsWith('/v1/messages')) {
      return inner(input, init)
    }

    const originalBody = await readAnthropicBody(input, init)
    const transformedBody = anthropicToOpenaiChat(originalBody)
    const headers = buildOpenAIChatCompatibleHeaders(init?.headers)
    const upstreamUrl = new URL(
      url.toString().replace(/\/v1\/messages(?:\?.*)?$/, '/v3/chat/completions'),
    )

    logForDebugging(
      `[API REQUEST] ${url.pathname} remapped_to=OpenAI/Chat upstream_path=${upstreamUrl.pathname} source=${source ?? 'unknown'} request_id=${randomUUID()}`,
    )

    const upstream = await inner(upstreamUrl.toString(), {
      method: init?.method ?? 'POST',
      headers,
      body: JSON.stringify(transformedBody),
      signal: init?.signal,
    })

    if (!upstream.ok) {
      const errorText = await upstream.text().catch(() => '')
      return Response.json(
        {
          type: 'error',
          error: {
            type: 'api_error',
            message: `OpenAI chat upstream returned HTTP ${upstream.status}: ${errorText.slice(0, 500)}`,
          },
        },
        { status: upstream.status },
      )
    }

    if (transformedBody.stream) {
      if (!upstream.body) {
        return Response.json(
          {
            type: 'error',
            error: {
              type: 'api_error',
              message: 'OpenAI chat upstream returned no body for stream',
            },
          },
          { status: 502 },
        )
      }

      return new Response(
        openaiChatStreamToAnthropic(upstream.body, originalBody.model),
        {
          status: 200,
          headers: {
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            Connection: 'keep-alive',
          },
        },
      )
    }

    const responseBody = await upstream.json()
    return Response.json(
      openaiChatToAnthropic(responseBody, originalBody.model),
    )
  }
}

function buildOpenAIChatCompatibleHeaders(inputHeaders?: HeadersInit): Headers {
  const originalHeaders = new Headers(inputHeaders)
  const headers = new Headers()

  for (const [name, value] of originalHeaders.entries()) {
    if (name.toLowerCase() === 'anthropic-version') {
      continue
    }
    headers.set(name, value)
  }

  headers.set('Content-Type', 'application/json')

  if (!headers.has('x-api-key')) {
    const apiKey = process.env.ANTHROPIC_API_KEY
    const authHeader = headers.get('Authorization')
    if (apiKey) {
      headers.set('x-api-key', apiKey)
    } else if (authHeader?.startsWith('Bearer ')) {
      headers.set('x-api-key', authHeader.slice('Bearer '.length))
    }
  }

  return headers
}

async function readAnthropicBody(
  input: RequestInfo | URL,
  init?: RequestInit,
): Promise<AnthropicRequest> {
  const directBody = init?.body

  if (typeof directBody === 'string') {
    return JSON.parse(directBody) as AnthropicRequest
  }

  if (directBody instanceof Uint8Array || directBody instanceof ArrayBuffer) {
    return JSON.parse(Buffer.from(directBody).toString('utf8')) as AnthropicRequest
  }

  if (input instanceof Request) {
    return (await input.clone().json()) as AnthropicRequest
  }

  throw new Error('Unable to read Anthropic request body for OpenAI/Codex transformation')
}
