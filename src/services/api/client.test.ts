import { describe, expect, mock, test } from 'bun:test'

mock.module('src/utils/http.js', () => ({
  getAuthHeaders: mock(() => ({})),
  getMCPUserAgent: mock(() => 'client-test-agent'),
  getUserAgent: mock(() => 'client-test-agent'),
  getWebFetchUserAgent: mock(() => 'client-test-agent'),
  withOAuth401Retry: mock(async <T>(fn: () => Promise<T>) => fn()),
}))

describe('resolveAnthropicClientApiKey', () => {
  test('does not inherit a local api key when a provider auth token is explicit', async () => {
    const { resolveAnthropicClientApiKey } = await import('./client.js')
    const getFallbackApiKey = mock(() => 'sk-keychain-fallback')

    const apiKey = resolveAnthropicClientApiKey({
      envAuthToken: 'provider-bearer-token',
      envApiKey: undefined,
      getFallbackApiKey,
    })

    expect(apiKey).toBeNull()
    expect(getFallbackApiKey).not.toHaveBeenCalled()
  })

  test('preserves an explicit api key when the caller opts into dual auth', async () => {
    const { resolveAnthropicClientApiKey } = await import('./client.js')
    const getFallbackApiKey = mock(() => 'sk-keychain-fallback')

    const apiKey = resolveAnthropicClientApiKey({
      explicitApiKey: 'sk-explicit-api-key',
      envAuthToken: 'provider-bearer-token',
      getFallbackApiKey,
    })

    expect(apiKey).toBe('sk-explicit-api-key')
    expect(getFallbackApiKey).not.toHaveBeenCalled()
  })

  test('falls back to the local api key when no provider auth token is present', async () => {
    const { resolveAnthropicClientApiKey } = await import('./client.js')
    const getFallbackApiKey = mock(() => 'sk-keychain-fallback')

    const apiKey = resolveAnthropicClientApiKey({
      envAuthToken: '',
      envApiKey: '',
      getFallbackApiKey,
    })

    expect(apiKey).toBe('sk-keychain-fallback')
    expect(getFallbackApiKey).toHaveBeenCalled()
  })
})

describe('getAnthropicClient', () => {
  test('passes bearer-token provider auth without an SDK api key', async () => {
    const { getAnthropicClient } = await import('./client.js')
    const originalAuthToken = process.env.ANTHROPIC_AUTH_TOKEN
    const originalApiKey = process.env.ANTHROPIC_API_KEY
    const originalSimple = process.env.CLAUDE_CODE_SIMPLE

    process.env.ANTHROPIC_AUTH_TOKEN = 'provider-bearer-token'
    process.env.CLAUDE_CODE_SIMPLE = '1'
    delete process.env.ANTHROPIC_API_KEY

    try {
      const client = await getAnthropicClient({
        maxRetries: 0,
        model: 'claude-sonnet-4-6',
      })

      expect(client.apiKey).toBeNull()
      expect(client._options.defaultHeaders).toMatchObject({
        Authorization: 'Bearer provider-bearer-token',
      })
    } finally {
      if (originalAuthToken === undefined) delete process.env.ANTHROPIC_AUTH_TOKEN
      else process.env.ANTHROPIC_AUTH_TOKEN = originalAuthToken

      if (originalApiKey === undefined) delete process.env.ANTHROPIC_API_KEY
      else process.env.ANTHROPIC_API_KEY = originalApiKey

      if (originalSimple === undefined) delete process.env.CLAUDE_CODE_SIMPLE
      else process.env.CLAUDE_CODE_SIMPLE = originalSimple
    }
  })

  test('rewrites Anthropic messages calls to Ark chat completions for /api base URLs', async () => {
    const { getAnthropicClient } = await import('./client.js')
    const originalAuthToken = process.env.ANTHROPIC_AUTH_TOKEN
    const originalApiKey = process.env.ANTHROPIC_API_KEY
    const originalBaseUrl = process.env.ANTHROPIC_BASE_URL
    const originalSimple = process.env.CLAUDE_CODE_SIMPLE

    process.env.ANTHROPIC_API_KEY = 'ark-test-key'
    process.env.ANTHROPIC_BASE_URL = 'https://ark-cn-beijing.bytedance.net/api'
    process.env.CLAUDE_CODE_SIMPLE = '1'
    delete process.env.ANTHROPIC_AUTH_TOKEN

    const seen: { url: string; body: Record<string, unknown> }[] = []
    const fetchMock = mock(async (input: RequestInfo | URL, init?: RequestInit) => {
      seen.push({
        url: input instanceof Request ? input.url : String(input),
        body: JSON.parse(String(init?.body ?? '{}')) as Record<string, unknown>,
      })

      return Response.json({
        id: 'chatcmpl-1',
        object: 'chat.completion',
        created: 1,
        model: 'ep-ark-model',
        choices: [
          {
            index: 0,
            message: { role: 'assistant', content: 'hello from ark' },
            finish_reason: 'stop',
          },
        ],
        usage: {
          prompt_tokens: 12,
          completion_tokens: 4,
          total_tokens: 16,
        },
      })
    })

    try {
      const client = await getAnthropicClient({
        maxRetries: 0,
        model: 'ep-ark-model',
        fetchOverride: fetchMock,
      })

      const response = await client.beta.messages.create({
        model: 'ep-ark-model',
        max_tokens: 16,
        messages: [{ role: 'user', content: 'ping' }],
      })

      expect(seen).toHaveLength(1)
      expect(seen[0]?.url).toBe('https://ark-cn-beijing.bytedance.net/api/v3/chat/completions')
      expect(seen[0]?.body).toMatchObject({
        model: 'ep-ark-model',
        messages: [{ role: 'user', content: 'ping' }],
      })
      expect(response.type).toBe('message')
      expect(response.content).toEqual([{ type: 'text', text: 'hello from ark' }])
    } finally {
      if (originalAuthToken === undefined) delete process.env.ANTHROPIC_AUTH_TOKEN
      else process.env.ANTHROPIC_AUTH_TOKEN = originalAuthToken

      if (originalApiKey === undefined) delete process.env.ANTHROPIC_API_KEY
      else process.env.ANTHROPIC_API_KEY = originalApiKey

      if (originalBaseUrl === undefined) delete process.env.ANTHROPIC_BASE_URL
      else process.env.ANTHROPIC_BASE_URL = originalBaseUrl

      if (originalSimple === undefined) delete process.env.CLAUDE_CODE_SIMPLE
      else process.env.CLAUDE_CODE_SIMPLE = originalSimple
    }
  })
})

describe('buildOpenAIChatCompatibleFetch', () => {
  test('transforms OpenAI chat SSE responses back into Anthropic SSE', async () => {
    const { buildOpenAIChatCompatibleFetch } = await import('../openaiAuth/fetch.js')

    const fetchMock = mock(async (input: RequestInfo | URL) => {
      const upstreamText = [
        'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"ep-ark-model","choices":[{"index":0,"delta":{"role":"assistant","content":"hello"},"finish_reason":null}]}',
        '',
        'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"ep-ark-model","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":"stop"}],"usage":{"prompt_tokens":12,"completion_tokens":4,"total_tokens":16}}',
        '',
        'data: [DONE]',
        '',
      ].join('\n')

      return new Response(upstreamText, {
        status: 200,
        headers: { 'Content-Type': 'text/event-stream' },
      })
    })

    const compatFetch = buildOpenAIChatCompatibleFetch(fetchMock, 'client-test')
    const response = await compatFetch(
      'https://ark-cn-beijing.bytedance.net/api/v1/messages?beta=true',
      {
        method: 'POST',
        headers: {
          Authorization: 'Bearer ark-test-key',
          'anthropic-version': '2023-06-01',
        },
        body: JSON.stringify({
          model: 'ep-ark-model',
          max_tokens: 16,
          stream: true,
          messages: [{ role: 'user', content: 'ping' }],
        }),
      },
    )

    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(String(fetchMock.mock.calls[0]?.[0])).toBe(
      'https://ark-cn-beijing.bytedance.net/api/v3/chat/completions',
    )
    expect(response.headers.get('Content-Type')).toBe('text/event-stream')

    const sseText = await new Response(response.body).text()
    expect(sseText).toContain('event: message_start')
    expect(sseText).toContain('event: content_block_delta')
    expect(sseText).toContain('hello')
    expect(sseText).toContain(' world')
    expect(sseText).toContain('event: message_stop')
  })
})
