/**
 * @fileoverview GitHub Copilot adapter implementing {@link LLMAdapter}.
 *
 * Uses the OpenAI-compatible Copilot Chat Completions endpoint at
 * `https://api.githubcopilot.com`. Authentication requires a GitHub token
 * (e.g. from `gh auth token`) which is exchanged for a short-lived Copilot
 * session token via the internal token endpoint.
 *
 * API key resolution order:
 *   1. `apiKey` constructor argument
 *   2. `GITHUB_TOKEN` environment variable
 *
 * @example
 * ```ts
 * import { CopilotAdapter } from './copilot.js'
 *
 * const adapter = new CopilotAdapter()          // uses GITHUB_TOKEN env var
 * const response = await adapter.chat(messages, {
 *   model: 'claude-sonnet-4',
 *   maxTokens: 4096,
 * })
 * ```
 */

import OpenAI from 'openai'
import type {
  ChatCompletion,
  ChatCompletionAssistantMessageParam,
  ChatCompletionChunk,
  ChatCompletionMessageParam,
  ChatCompletionMessageToolCall,
  ChatCompletionTool,
  ChatCompletionToolMessageParam,
  ChatCompletionUserMessageParam,
} from 'openai/resources/chat/completions/index.js'

import type {
  ContentBlock,
  LLMAdapter,
  LLMChatOptions,
  LLMMessage,
  LLMResponse,
  LLMStreamOptions,
  LLMToolDef,
  StreamEvent,
  TextBlock,
  ToolUseBlock,
} from '../types.js'

// ---------------------------------------------------------------------------
// Copilot auth — OAuth2 device flow + token exchange
// ---------------------------------------------------------------------------

const COPILOT_TOKEN_URL = 'https://api.github.com/copilot_internal/v2/token'
const DEVICE_CODE_URL   = 'https://github.com/login/device/code'
const POLL_URL          = 'https://github.com/login/oauth/access_token'
const COPILOT_CLIENT_ID = 'Iv1.b507a08c87ecfe98'

const COPILOT_HEADERS: Record<string, string> = {
  'Copilot-Integration-Id': 'vscode-chat',
  'Editor-Version': 'vscode/1.100.0',
  'Editor-Plugin-Version': 'copilot-chat/0.42.2',
}

interface CopilotTokenResponse {
  token: string
  expires_at: number
}

interface DeviceCodeResponse {
  device_code: string
  user_code: string
  verification_uri: string
  interval: number
  expires_in: number
}

interface PollResponse {
  access_token?: string
  error?: string
  error_description?: string
}

/**
 * Start the GitHub OAuth2 device code flow with the Copilot client ID.
 *
 * Prints a user code and URL to stdout, then polls until the user completes
 * authorization in their browser. Returns a GitHub OAuth token scoped for
 * Copilot access.
 */
async function deviceCodeLogin(): Promise<string> {
  // Step 1: Request a device code
  const codeRes = await fetch(DEVICE_CODE_URL, {
    method: 'POST',
    headers: {
      Accept: 'application/json',
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ client_id: COPILOT_CLIENT_ID, scope: 'copilot' }),
  })

  if (!codeRes.ok) {
    const body = await codeRes.text().catch(() => '')
    throw new Error(`Device code request failed (${codeRes.status}): ${body}`)
  }

  const codeData = (await codeRes.json()) as DeviceCodeResponse

  // Step 2: Prompt the user
  console.log(`\n┌─────────────────────────────────────────────┐`)
  console.log(`│  GitHub Copilot — Sign in                    │`)
  console.log(`│                                              │`)
  console.log(`│  Open:  ${codeData.verification_uri.padEnd(35)}│`)
  console.log(`│  Code:  ${codeData.user_code.padEnd(35)}│`)
  console.log(`└─────────────────────────────────────────────┘\n`)

  // Step 3: Poll for the user to complete auth
  const interval = (codeData.interval || 5) * 1000
  const deadline = Date.now() + codeData.expires_in * 1000

  while (Date.now() < deadline) {
    await new Promise((resolve) => setTimeout(resolve, interval))

    const pollRes = await fetch(POLL_URL, {
      method: 'POST',
      headers: {
        Accept: 'application/json',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        client_id: COPILOT_CLIENT_ID,
        device_code: codeData.device_code,
        grant_type: 'urn:ietf:params:oauth:grant-type:device_code',
      }),
    })

    const pollData = (await pollRes.json()) as PollResponse

    if (pollData.access_token) {
      console.log('✓ Authenticated with GitHub Copilot\n')
      return pollData.access_token
    }

    if (pollData.error === 'authorization_pending') continue
    if (pollData.error === 'slow_down') {
      await new Promise((resolve) => setTimeout(resolve, 5000))
      continue
    }

    throw new Error(
      `OAuth device flow failed: ${pollData.error} — ${pollData.error_description ?? ''}`,
    )
  }

  throw new Error('Device code expired. Please try again.')
}

/**
 * Exchange a GitHub OAuth token (from the Copilot device flow) for a
 * short-lived Copilot session token.
 *
 * Note: the token exchange endpoint does NOT require the Copilot-specific
 * headers (Editor-Version etc.) — only the chat completions endpoint does.
 */
async function fetchCopilotToken(githubToken: string): Promise<CopilotTokenResponse> {
  const res = await fetch(COPILOT_TOKEN_URL, {
    method: 'GET',
    headers: {
      Authorization: `token ${githubToken}`,
      Accept: 'application/json',
      'User-Agent': 'GitHubCopilotChat/0.28.0',
    },
  })

  if (!res.ok) {
    const body = await res.text().catch(() => '')
    throw new Error(
      `Copilot token exchange failed (${res.status}): ${body || res.statusText}`,
    )
  }

  return (await res.json()) as CopilotTokenResponse
}

// ---------------------------------------------------------------------------
// Internal helpers — framework → OpenAI (shared with openai.ts pattern)
// ---------------------------------------------------------------------------

function toOpenAITool(tool: LLMToolDef): ChatCompletionTool {
  return {
    type: 'function',
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.inputSchema as Record<string, unknown>,
    },
  }
}

function hasToolResults(msg: LLMMessage): boolean {
  return msg.content.some((b) => b.type === 'tool_result')
}

function toOpenAIMessages(messages: LLMMessage[]): ChatCompletionMessageParam[] {
  const result: ChatCompletionMessageParam[] = []

  for (const msg of messages) {
    if (msg.role === 'assistant') {
      result.push(toOpenAIAssistantMessage(msg))
    } else {
      if (!hasToolResults(msg)) {
        result.push(toOpenAIUserMessage(msg))
      } else {
        const nonToolBlocks = msg.content.filter((b) => b.type !== 'tool_result')
        if (nonToolBlocks.length > 0) {
          result.push(toOpenAIUserMessage({ role: 'user', content: nonToolBlocks }))
        }
        for (const block of msg.content) {
          if (block.type === 'tool_result') {
            const toolMsg: ChatCompletionToolMessageParam = {
              role: 'tool',
              tool_call_id: block.tool_use_id,
              content: block.content,
            }
            result.push(toolMsg)
          }
        }
      }
    }
  }

  return result
}

function toOpenAIUserMessage(msg: LLMMessage): ChatCompletionUserMessageParam {
  if (msg.content.length === 1 && msg.content[0]?.type === 'text') {
    return { role: 'user', content: msg.content[0].text }
  }

  type ContentPart = OpenAI.Chat.ChatCompletionContentPartText | OpenAI.Chat.ChatCompletionContentPartImage
  const parts: ContentPart[] = []

  for (const block of msg.content) {
    if (block.type === 'text') {
      parts.push({ type: 'text', text: block.text })
    } else if (block.type === 'image') {
      parts.push({
        type: 'image_url',
        image_url: {
          url: `data:${block.source.media_type};base64,${block.source.data}`,
        },
      })
    }
  }

  return { role: 'user', content: parts }
}

function toOpenAIAssistantMessage(msg: LLMMessage): ChatCompletionAssistantMessageParam {
  const toolCalls: ChatCompletionMessageToolCall[] = []
  const textParts: string[] = []

  for (const block of msg.content) {
    if (block.type === 'tool_use') {
      toolCalls.push({
        id: block.id,
        type: 'function',
        function: {
          name: block.name,
          arguments: JSON.stringify(block.input),
        },
      })
    } else if (block.type === 'text') {
      textParts.push(block.text)
    }
  }

  const assistantMsg: ChatCompletionAssistantMessageParam = {
    role: 'assistant',
    content: textParts.length > 0 ? textParts.join('') : null,
  }

  if (toolCalls.length > 0) {
    assistantMsg.tool_calls = toolCalls
  }

  return assistantMsg
}

// ---------------------------------------------------------------------------
// Internal helpers — OpenAI → framework
// ---------------------------------------------------------------------------

function fromOpenAICompletion(completion: ChatCompletion): LLMResponse {
  const choice = completion.choices[0]
  if (choice === undefined) {
    throw new Error('Copilot returned a completion with no choices')
  }

  const content: ContentBlock[] = []
  const message = choice.message

  if (message.content !== null && message.content !== undefined) {
    const textBlock: TextBlock = { type: 'text', text: message.content }
    content.push(textBlock)
  }

  for (const toolCall of message.tool_calls ?? []) {
    let parsedInput: Record<string, unknown> = {}
    try {
      const parsed: unknown = JSON.parse(toolCall.function.arguments)
      if (parsed !== null && typeof parsed === 'object' && !Array.isArray(parsed)) {
        parsedInput = parsed as Record<string, unknown>
      }
    } catch {
      // Malformed arguments — surface as empty object.
    }

    const toolUseBlock: ToolUseBlock = {
      type: 'tool_use',
      id: toolCall.id,
      name: toolCall.function.name,
      input: parsedInput,
    }
    content.push(toolUseBlock)
  }

  const stopReason = normalizeFinishReason(choice.finish_reason ?? 'stop')

  return {
    id: completion.id,
    content,
    model: completion.model,
    stop_reason: stopReason,
    usage: {
      input_tokens: completion.usage?.prompt_tokens ?? 0,
      output_tokens: completion.usage?.completion_tokens ?? 0,
    },
  }
}

function normalizeFinishReason(reason: string): string {
  switch (reason) {
    case 'stop':           return 'end_turn'
    case 'tool_calls':     return 'tool_use'
    case 'length':         return 'max_tokens'
    case 'content_filter': return 'content_filter'
    default:               return reason
  }
}

// ---------------------------------------------------------------------------
// Adapter implementation
// ---------------------------------------------------------------------------

/**
 * LLM adapter backed by the GitHub Copilot Chat Completions API.
 *
 * Authentication options (tried in order):
 *   1. `apiKey` constructor arg — a GitHub OAuth token already scoped for Copilot
 *   2. `GITHUB_COPILOT_TOKEN` env var — same as above
 *   3. Interactive OAuth2 device flow — prompts the user to sign in via browser
 *
 * The GitHub token is exchanged for a short-lived Copilot session token, which
 * is cached and auto-refreshed.
 *
 * Thread-safe — a single instance may be shared across concurrent agent runs.
 */
export class CopilotAdapter implements LLMAdapter {
  readonly name = 'copilot'

  #githubToken: string | null
  #cachedToken: string | null = null
  #tokenExpiresAt = 0

  constructor(apiKey?: string) {
    this.#githubToken = apiKey
      ?? process.env['GITHUB_COPILOT_TOKEN']
      ?? process.env['GITHUB_TOKEN']
      ?? null
  }

  /**
   * Return a valid Copilot session token, refreshing if necessary.
   * If no GitHub token is available, triggers the interactive device flow.
   */
  async #getSessionToken(): Promise<string> {
    const now = Math.floor(Date.now() / 1000)
    if (this.#cachedToken && this.#tokenExpiresAt - 60 > now) {
      return this.#cachedToken
    }

    // If we don't have a GitHub token yet, do the device flow
    if (!this.#githubToken) {
      this.#githubToken = await deviceCodeLogin()
    }

    const resp = await fetchCopilotToken(this.#githubToken)
    this.#cachedToken = resp.token
    this.#tokenExpiresAt = resp.expires_at
    return resp.token
  }

  /** Build a short-lived OpenAI client pointed at the Copilot endpoint. */
  async #createClient(): Promise<OpenAI> {
    const sessionToken = await this.#getSessionToken()
    return new OpenAI({
      apiKey: sessionToken,
      baseURL: 'https://api.githubcopilot.com',
      defaultHeaders: COPILOT_HEADERS,
    })
  }

  // -------------------------------------------------------------------------
  // chat()
  // -------------------------------------------------------------------------

  async chat(messages: LLMMessage[], options: LLMChatOptions): Promise<LLMResponse> {
    const client = await this.#createClient()
    const openAIMessages = buildOpenAIMessageList(messages, options.systemPrompt)

    const completion = await client.chat.completions.create(
      {
        model: options.model,
        messages: openAIMessages,
        max_tokens: options.maxTokens,
        temperature: options.temperature,
        tools: options.tools ? options.tools.map(toOpenAITool) : undefined,
        stream: false,
      },
      {
        signal: options.abortSignal,
      },
    )

    return fromOpenAICompletion(completion)
  }

  // -------------------------------------------------------------------------
  // stream()
  // -------------------------------------------------------------------------

  async *stream(
    messages: LLMMessage[],
    options: LLMStreamOptions,
  ): AsyncIterable<StreamEvent> {
    const client = await this.#createClient()
    const openAIMessages = buildOpenAIMessageList(messages, options.systemPrompt)

    const streamResponse = await client.chat.completions.create(
      {
        model: options.model,
        messages: openAIMessages,
        max_tokens: options.maxTokens,
        temperature: options.temperature,
        tools: options.tools ? options.tools.map(toOpenAITool) : undefined,
        stream: true,
        stream_options: { include_usage: true },
      },
      {
        signal: options.abortSignal,
      },
    )

    let completionId = ''
    let completionModel = ''
    let finalFinishReason: string = 'stop'
    let inputTokens = 0
    let outputTokens = 0
    const toolCallBuffers = new Map<
      number,
      { id: string; name: string; argsJson: string }
    >()
    let fullText = ''

    try {
      for await (const chunk of streamResponse) {
        completionId = chunk.id
        completionModel = chunk.model

        if (chunk.usage !== null && chunk.usage !== undefined) {
          inputTokens = chunk.usage.prompt_tokens
          outputTokens = chunk.usage.completion_tokens
        }

        const choice: ChatCompletionChunk.Choice | undefined = chunk.choices[0]
        if (choice === undefined) continue

        const delta = choice.delta

        if (delta.content !== null && delta.content !== undefined) {
          fullText += delta.content
          const textEvent: StreamEvent = { type: 'text', data: delta.content }
          yield textEvent
        }

        for (const toolCallDelta of delta.tool_calls ?? []) {
          const idx = toolCallDelta.index

          if (!toolCallBuffers.has(idx)) {
            toolCallBuffers.set(idx, {
              id: toolCallDelta.id ?? '',
              name: toolCallDelta.function?.name ?? '',
              argsJson: '',
            })
          }

          const buf = toolCallBuffers.get(idx)
          if (buf !== undefined) {
            if (toolCallDelta.id) buf.id = toolCallDelta.id
            if (toolCallDelta.function?.name) buf.name = toolCallDelta.function.name
            if (toolCallDelta.function?.arguments) {
              buf.argsJson += toolCallDelta.function.arguments
            }
          }
        }

        if (choice.finish_reason !== null && choice.finish_reason !== undefined) {
          finalFinishReason = choice.finish_reason
        }
      }

      const finalToolUseBlocks: ToolUseBlock[] = []
      for (const buf of toolCallBuffers.values()) {
        let parsedInput: Record<string, unknown> = {}
        try {
          const parsed: unknown = JSON.parse(buf.argsJson)
          if (parsed !== null && typeof parsed === 'object' && !Array.isArray(parsed)) {
            parsedInput = parsed as Record<string, unknown>
          }
        } catch {
          // Malformed JSON — surface as empty object.
        }

        const toolUseBlock: ToolUseBlock = {
          type: 'tool_use',
          id: buf.id,
          name: buf.name,
          input: parsedInput,
        }
        finalToolUseBlocks.push(toolUseBlock)
        const toolUseEvent: StreamEvent = { type: 'tool_use', data: toolUseBlock }
        yield toolUseEvent
      }

      const doneContent: ContentBlock[] = []
      if (fullText.length > 0) {
        const textBlock: TextBlock = { type: 'text', text: fullText }
        doneContent.push(textBlock)
      }
      doneContent.push(...finalToolUseBlocks)

      const finalResponse: LLMResponse = {
        id: completionId,
        content: doneContent,
        model: completionModel,
        stop_reason: normalizeFinishReason(finalFinishReason),
        usage: { input_tokens: inputTokens, output_tokens: outputTokens },
      }

      const doneEvent: StreamEvent = { type: 'done', data: finalResponse }
      yield doneEvent
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err))
      const errorEvent: StreamEvent = { type: 'error', data: error }
      yield errorEvent
    }
  }
}

// ---------------------------------------------------------------------------
// Private utility
// ---------------------------------------------------------------------------

function buildOpenAIMessageList(
  messages: LLMMessage[],
  systemPrompt: string | undefined,
): ChatCompletionMessageParam[] {
  const result: ChatCompletionMessageParam[] = []

  if (systemPrompt !== undefined && systemPrompt.length > 0) {
    result.push({ role: 'system', content: systemPrompt })
  }

  result.push(...toOpenAIMessages(messages))
  return result
}

// Re-export types that consumers of this module commonly need alongside the adapter.
export type {
  ContentBlock,
  LLMAdapter,
  LLMChatOptions,
  LLMMessage,
  LLMResponse,
  LLMStreamOptions,
  LLMToolDef,
  StreamEvent,
}

// ---------------------------------------------------------------------------
// Premium request multipliers
// ---------------------------------------------------------------------------

/**
 * Model metadata used for display names, context windows, and premium request
 * multiplier lookup.
 */
export interface CopilotModelInfo {
  readonly id: string
  readonly name: string
  readonly contextWindow: number
}

/**
 * Return the premium-request multiplier for a Copilot model.
 *
 * Copilot doesn't charge per-token — instead each request costs
 * `multiplier × 1 premium request` from the user's monthly allowance.
 * A multiplier of 0 means the model is included at no premium cost.
 *
 * Based on https://docs.github.com/en/copilot/reference/ai-models/supported-models#model-multipliers
 */
export function getCopilotMultiplier(modelId: string): number {
  const id = modelId.toLowerCase()

  // 0x — included models
  if (id.includes('gpt-4.1'))   return 0
  if (id.includes('gpt-4o'))    return 0
  if (id.includes('gpt-5-mini') || id.includes('gpt-5 mini')) return 0
  if (id.includes('raptor'))    return 0
  if (id.includes('goldeneye')) return 0

  // 0.25x
  if (id.includes('grok'))      return 0.25

  // 0.33x
  if (id.includes('claude-haiku'))                             return 0.33
  if (id.includes('gemini-3-flash') || id.includes('gemini-3.0-flash')) return 0.33
  if (id.includes('gpt-5.1-codex-mini'))                      return 0.33
  if (id.includes('gpt-5.4-mini') || id.includes('gpt-5.4 mini')) return 0.33

  // 1x — standard premium
  if (id.includes('claude-sonnet'))  return 1
  if (id.includes('gemini-2.5-pro')) return 1
  if (id.includes('gemini-3-pro') || id.includes('gemini-3.0-pro')) return 1
  if (id.includes('gemini-3.1-pro')) return 1
  if (id.includes('gpt-5.1'))       return 1
  if (id.includes('gpt-5.2'))       return 1
  if (id.includes('gpt-5.3'))       return 1
  if (id.includes('gpt-5.4'))       return 1

  // 30x — fast opus
  if (id.includes('claude-opus') && id.includes('fast')) return 30

  // 3x — opus
  if (id.includes('claude-opus'))    return 3

  return 1
}

/**
 * Human-readable string describing the premium-request cost for a model.
 *
 * Examples: `"included (0×)"`, `"1× premium request"`, `"0.33× premium request"`
 */
export function formatCopilotMultiplier(multiplier: number): string {
  if (multiplier === 0) return 'included (0×)'
  if (Number.isInteger(multiplier)) return `${multiplier}× premium request`
  return `${multiplier}× premium request`
}

/** Known model metadata for Copilot-available models. */
export const COPILOT_MODELS: readonly CopilotModelInfo[] = [
  { id: 'gpt-4.1',             name: 'GPT-4.1',                  contextWindow: 128_000  },
  { id: 'gpt-4o',              name: 'GPT-4o',                   contextWindow: 128_000  },
  { id: 'gpt-5-mini',          name: 'GPT-5 mini',               contextWindow: 200_000  },
  { id: 'gpt-5.1',             name: 'GPT-5.1',                  contextWindow: 200_000  },
  { id: 'gpt-5.1-codex',       name: 'GPT-5.1-Codex',            contextWindow: 200_000  },
  { id: 'gpt-5.1-codex-mini',  name: 'GPT-5.1-Codex-Mini',       contextWindow: 200_000  },
  { id: 'gpt-5.1-codex-max',   name: 'GPT-5.1-Codex-Max',        contextWindow: 200_000  },
  { id: 'gpt-5.2',             name: 'GPT-5.2',                  contextWindow: 200_000  },
  { id: 'gpt-5.2-codex',       name: 'GPT-5.2-Codex',            contextWindow: 200_000  },
  { id: 'gpt-5.3-codex',       name: 'GPT-5.3-Codex',            contextWindow: 200_000  },
  { id: 'gpt-5.4',             name: 'GPT-5.4',                  contextWindow: 200_000  },
  { id: 'gpt-5.4-mini',        name: 'GPT-5.4 mini',             contextWindow: 200_000  },
  { id: 'claude-haiku-4.5',    name: 'Claude Haiku 4.5',          contextWindow: 200_000  },
  { id: 'claude-opus-4.5',     name: 'Claude Opus 4.5',           contextWindow: 200_000  },
  { id: 'claude-opus-4.6',     name: 'Claude Opus 4.6',           contextWindow: 200_000  },
  { id: 'claude-opus-4.6-fast', name: 'Claude Opus 4.6 (fast)',   contextWindow: 200_000  },
  { id: 'claude-sonnet-4',     name: 'Claude Sonnet 4',           contextWindow: 200_000  },
  { id: 'claude-sonnet-4.5',   name: 'Claude Sonnet 4.5',         contextWindow: 200_000  },
  { id: 'claude-sonnet-4.6',   name: 'Claude Sonnet 4.6',         contextWindow: 200_000  },
  { id: 'gemini-2.5-pro',      name: 'Gemini 2.5 Pro',            contextWindow: 1_000_000 },
  { id: 'gemini-3-flash',      name: 'Gemini 3 Flash',            contextWindow: 1_000_000 },
  { id: 'gemini-3-pro',        name: 'Gemini 3 Pro',              contextWindow: 1_000_000 },
  { id: 'gemini-3.1-pro',      name: 'Gemini 3.1 Pro',            contextWindow: 1_000_000 },
  { id: 'grok-code-fast-1',    name: 'Grok Code Fast 1',          contextWindow: 128_000  },
  { id: 'raptor-mini',         name: 'Raptor mini',               contextWindow: 128_000  },
  { id: 'goldeneye',           name: 'Goldeneye',                 contextWindow: 128_000  },
] as const
