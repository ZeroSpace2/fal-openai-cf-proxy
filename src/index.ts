/**
 * Welcome to Cloudflare Workers! This is your first worker.
 *
 * - Run `npm run dev` in your terminal to start a development server
 * - Open a browser tab at http://localhost:8787/ to see your worker in action
 * - Run `npm run deploy` to publish your worker
 *
 * Bind resources to your worker in `wrangler.jsonc`. After adding bindings, a type definition for the
 * `Env` object can be regenerated with `npm run cf-typegen`.
 *
 * Learn more at https://developers.cloudflare.com/workers/
 */

import { fal } from '@fal-ai/client';

// Define the expected environment variables/secrets
interface Env {
	/**
	 * Comma-separated list of Fal AI API keys for redundancy.
	 * Example: "key1,key2,key3"
	 * Configure this secret in your Cloudflare dashboard or wrangler.jsonc.
	 */
	FAL_KEY: string;
	/**
	 * Comma-separated list of allowed API keys for client authentication.
	 * Example: "clientkey1,clientkey2,verysecretkey"
	 * Configure this secret in your Cloudflare dashboard or wrangler.jsonc.
	 */
	API_KEY: string;
	/**
	 * Comma-separated list of allowed origins for CORS.
	 * Example: "https://example.com,https://app.example.com"
	 * If not set, defaults to '*' (all origins).
	 * Configure this secret in your Cloudflare dashboard or wrangler.jsonc.
	 */
	ALLOWED_ORIGINS?: string;
}

// OpenAI Request/Response Types (Simplified)
interface OpenAIMessage {
	role: 'system' | 'user' | 'assistant';
	content: string | null;
}

interface OpenAIChatCompletionRequest {
	model: string;
	messages: OpenAIMessage[];
	stream?: boolean;
	// We can add other OpenAI parameters here if needed, but keep them simple for now
	// e.g., max_tokens, temperature, etc.
	reasoning_effort?: 'low' | 'medium' | 'high'; // New param compatible with OpenAI reasoning
}

// Type for supported Fal Model IDs based on the constant array
// This ensures type safety when passing to the fal client
type FalModelId = typeof FAL_SUPPORTED_MODELS[number];

interface FalInput {
	model: FalModelId; // Use the specific type
	prompt: string;
	system_prompt?: string;
	reasoning?: boolean;
	// Fal-specific parameters can be added here if needed
}

// Payload for fal.stream or fal.subscribe
interface FalPayload {
    input: FalInput;
    // Add other stream/subscribe options if needed
}

// Simple type for log entries in the stream
interface LogEntry {
	message: string;
	level?: string; // Optional level property
}

// Type for Fal Stream Event Data
interface FalStreamEventData {
    output: string;
    reasoning?: any; // Or a more specific type if known
    partial?: boolean;
    error?: any; // Or a more specific type if known
    logs?: LogEntry[];
}

// Type for Fal Stream Event (passed in iterator)
interface FalStreamEvent {
	output: string;
    reasoning?: any; // Or a more specific type if known
    partial?: boolean;
    error?: any; // Or a more specific type if known
    logs?: LogEntry[];
	// Other potential properties from the stream event can be added here
}

// Type for Fal Subscribe Result
interface FalSubscribeResult {
	data: FalStreamEventData; // Assuming subscribe result structure matches stream event data
	requestId: string;
	// Other potential top-level properties can be added here
}

// === Global Definitions ===
const PROMPT_LIMIT = 4800;
const SYSTEM_PROMPT_LIMIT = 4800;

const FAL_SUPPORTED_MODELS = [
	"anthropic/claude-3.7-sonnet",
	"anthropic/claude-3.5-sonnet",
	"anthropic/claude-3-5-haiku",
	"anthropic/claude-3-haiku",
	"google/gemini-pro-1.5",
	"google/gemini-flash-1.5",
	"google/gemini-flash-1.5-8b",
	"google/gemini-2.0-flash-001",
	"meta-llama/llama-3.2-1b-instruct",
	"meta-llama/llama-3.2-3b-instruct",
	"meta-llama/llama-3.1-8b-instruct",
	"meta-llama/llama-3.1-70b-instruct",
	"openai/gpt-4o-mini",
	"openai/gpt-4o",
	"deepseek/deepseek-r1",
	"meta-llama/llama-4-maverick",
	"meta-llama/llama-4-scout"
] as const; // Use 'as const' for stricter type checking

// Helper function to get owner from model ID
const getOwner = (modelId: string): string => {
	if (modelId && modelId.includes('/')) {
		return modelId.split('/')[0];
	}
	return 'fal-ai'; // Default owner
};

// === Message Conversion Logic (Adapted from JS) ===
function convertMessagesToFalPrompt(messages: OpenAIMessage[]): { system_prompt: string; prompt: string } {
	let fixed_system_prompt_content = "";
	const conversation_message_blocks: string[] = [];
	// console.log(`Original messages count: ${messages.length}`); // Reduce noise

	for (const message of messages) {
		let content = (message.content === null || message.content === undefined) ? "" : String(message.content);
		switch (message.role) {
			case 'system':
				fixed_system_prompt_content += `System: ${content}\n\n`;
				break;
			case 'user':
				conversation_message_blocks.push(`Human: ${content}\n\n`);
				break;
			case 'assistant':
				conversation_message_blocks.push(`Assistant: ${content}\n\n`);
				break;
			default:
				console.warn(`Unsupported role: ${message.role}`);
				continue;
		}
	}

	if (fixed_system_prompt_content.length > SYSTEM_PROMPT_LIMIT) {
		const originalLength = fixed_system_prompt_content.length;
		fixed_system_prompt_content = fixed_system_prompt_content.substring(0, SYSTEM_PROMPT_LIMIT);
		console.warn(`Combined system messages truncated from ${originalLength} to ${SYSTEM_PROMPT_LIMIT}`);
	}
	fixed_system_prompt_content = fixed_system_prompt_content.trim();

	let space_occupied_by_fixed_system = 0;
	if (fixed_system_prompt_content.length > 0) {
		space_occupied_by_fixed_system = fixed_system_prompt_content.length + 4; // Approx space for separator
	}
	const remaining_system_limit = Math.max(0, SYSTEM_PROMPT_LIMIT - space_occupied_by_fixed_system);
	// console.log(`Trimmed fixed system prompt length: ${fixed_system_prompt_content.length}. Approx remaining system history limit: ${remaining_system_limit}`); // Reduce noise

	const prompt_history_blocks: string[] = [];
	const system_prompt_history_blocks: string[] = [];
	let current_prompt_length = 0;
	let current_system_history_length = 0;
	let promptFull = false;
	let systemHistoryFull = (remaining_system_limit <= 0);

	// console.log(`Processing ${conversation_message_blocks.length} user/assistant messages for recency filling.`); // Reduce noise
	for (let i = conversation_message_blocks.length - 1; i >= 0; i--) {
		const message_block = conversation_message_blocks[i];
		const block_length = message_block.length;

		if (promptFull && systemHistoryFull) {
			// console.log(`Both prompt and system history slots full. Omitting older messages from index ${i}.`); // Reduce noise
			break;
		}

		if (!promptFull) {
			if (current_prompt_length + block_length <= PROMPT_LIMIT) {
				prompt_history_blocks.unshift(message_block);
				current_prompt_length += block_length;
				continue;
			} else {
				promptFull = true;
				// console.log(`Prompt limit (${PROMPT_LIMIT}) reached. Trying system history slot.`); // Reduce noise
			}
		}

		if (!systemHistoryFull) {
			if (current_system_history_length + block_length <= remaining_system_limit) {
				 system_prompt_history_blocks.unshift(message_block);
				 current_system_history_length += block_length;
				 continue;
			} else {
				 systemHistoryFull = true;
				 // console.log(`System history limit (${remaining_system_limit}) reached.`); // Reduce noise
			}
		}
	}

	const system_prompt_history_content = system_prompt_history_blocks.join('').trim();
	const final_prompt = prompt_history_blocks.join('').trim();
	const SEPARATOR = "\n\n-------下面是比较早之前的对话内容-----\n\n";
	let final_system_prompt = "";
	const hasFixedSystem = fixed_system_prompt_content.length > 0;
	const hasSystemHistory = system_prompt_history_content.length > 0;

	if (hasFixedSystem && hasSystemHistory) {
		final_system_prompt = fixed_system_prompt_content + SEPARATOR + system_prompt_history_content;
		// console.log("Combining fixed system prompt and history with separator."); // Reduce noise
	} else if (hasFixedSystem) {
		final_system_prompt = fixed_system_prompt_content;
		// console.log("Using only fixed system prompt."); // Reduce noise
	} else if (hasSystemHistory) {
		final_system_prompt = system_prompt_history_content;
		// console.log("Using only history in system prompt slot."); // Reduce noise
	}

	const result = {
		system_prompt: final_system_prompt,
		prompt: final_prompt
	};

	// console.log(`Final system_prompt length: ${result.system_prompt.length}`); // Reduce noise
	// console.log(`Final prompt length: ${result.prompt.length}`); // Reduce noise
	return result;
}

// Type guard to check if a string is a valid FalModelId
function isValidFalModelId(modelId: string): modelId is FalModelId {
	return (FAL_SUPPORTED_MODELS as readonly string[]).includes(modelId);
}

// === CORS Utilities ===
// Default CORS headers
function getCorsHeaders(): HeadersInit {
    return {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS, PUT, DELETE, PATCH',
        'Access-Control-Allow-Headers': '*',
        'Access-Control-Max-Age': '86400', // Cache preflight for 1 day
    };
}

// Create a response with CORS headers included
function createCorsResponse(body: string | null, options: ResponseInit): Response {
    const corsHeaders = getCorsHeaders();
    const headers = { ...options.headers, ...corsHeaders };
    
    return new Response(body, {
        ...options,
        headers
    });
}

/**
 * Attempts a Fal API request (stream or subscribe) with key rotation on failure.
 * @param falKeys Array of Fal API keys.
 * @param falPayload Payload for the fal request.
 * @param stream Whether to use streaming.
 * @param modelToReport The original model name requested by the user for reporting.
 * @param ctx ExecutionContext for streaming requests.
 * @returns For stream=true: { readable: ReadableStream }. For stream=false: FalSubscribeResult.
 * @throws Throws an error if all keys fail or if a non-retryable error occurs.
 */
async function tryFalRequest(
    falKeys: string[],
    falPayload: FalPayload,
    stream: boolean,
    modelToReport: string, // Pass the original model name
    ctx?: ExecutionContext
): Promise<{ readable: ReadableStream } | FalSubscribeResult> {
    let lastError: any = null;

    if (falKeys.length === 0) {
        throw new Error("No Fal API keys configured.");
    }

    for (let i = 0; i < falKeys.length; i++) {
        const key = falKeys[i];
        console.log(`Attempting Fal request with key index ${i}`);
        try {
            // Configure the client with the current key for this attempt
            fal.config({ credentials: key });

            if (stream && ctx) {
                // --- Stream Handling ---
                console.log(`Initiating fal.stream with key index ${i}... Payload:`, JSON.stringify(falPayload, null, 2));
                const falStream = await fal.stream("fal-ai/any-llm", falPayload as any); // Use the prepared payload object
                console.log(`fal.stream initiated successfully with key index ${i}.`);

                // Set up the response stream
                const { readable, writable } = new TransformStream();
                const writer = writable.getWriter();
                const encoder = new TextEncoder();

                // Start background processing for the stream events
                const streamProcessing = (async () => {
                    try {
                        let previousOutput = '';
                        for await (const event of falStream) {
                           // console.log(`Fal Stream Event (Key ${i}):`, event); // Verbose logging
                            const currentOutput = (event && typeof event.output === 'string') ? event.output : '';
                            const isPartial = (event && typeof event.partial === 'boolean') ? event.partial : true; // Assume partial if not specified
                            const errorInfo = (event && event.error) ? event.error : null;
                            const logs = (event && Array.isArray((event as any).logs)) ? (event as any).logs : [];
                            const reasoningData = (event && event.reasoning) ? event.reasoning : null;
                            logs.forEach((log: LogEntry) => console.log(`[Fal Log - Key ${i}] ${log.message}`));

                            if (errorInfo) {
                                console.error(`Error received in fal stream event (Key ${i}):`, errorInfo);
                                const errorChunk = { id: `chatcmpl-${Date.now()}-error`, object: "chat.completion.chunk", created: Math.floor(Date.now() / 1000), model: modelToReport, choices: [{ index: 0, delta: {}, finish_reason: "error", logprobs: null, message: { role: 'assistant', content: `Fal Stream Error: ${JSON.stringify(errorInfo)}` } }] };
                                await writer.write(encoder.encode(`data: ${JSON.stringify(errorChunk)}\n\n`));
                                throw new Error(`Fal Stream Error: ${JSON.stringify(errorInfo)}`); // Throw to stop processing this stream
                            }

                            let deltaContent = '';
                            if (currentOutput.startsWith(previousOutput)) {
                                deltaContent = currentOutput.substring(previousOutput.length);
                            } else if (currentOutput.length > 0) {
                                console.warn(`Fal stream output mismatch (Key ${i}). Sending full current output as delta.`);
                                deltaContent = currentOutput;
                                previousOutput = ''; // Reset
                            }
                            if (currentOutput.length > 0) {
                                previousOutput = currentOutput;
                            }

                            if (deltaContent || reasoningData || !isPartial) {
                                const choice = {
                                    index: 0,
                                    delta: { content: deltaContent },
                                    finish_reason: isPartial === false ? "stop" : null,
                                    logprobs: null,
                                    ...(reasoningData && { reasoning: [{ index: 0, content: reasoningData }] })
                                };
                                const openAIChunk = {
                                    id: `chatcmpl-${Date.now()}-${Math.random().toString(36).substring(2, 15)}`,
                                    object: "chat.completion.chunk",
                                    created: Math.floor(Date.now() / 1000),
                                    model: modelToReport, // Report the originally requested model
                                    choices: [choice]
                                };
                                await writer.write(encoder.encode(`data: ${JSON.stringify(openAIChunk)}\n\n`));
                            }
                        }
                        console.log(`Fal stream iteration finished successfully (Key ${i}).`);
                        await writer.write(encoder.encode(`data: [DONE]\n\n`));

                    } catch (streamError: any) {
                        console.error(`Error during fal stream processing (Key ${i}):`, streamError);
                         try {
                            // Attempt to write an error chunk to the client stream
                            const errorDetails = (streamError instanceof Error) ? streamError.message : JSON.stringify(streamError);
							const errorChunk = { error: { message: "Stream processing error", type: "proxy_error", param: null, code: null, details: errorDetails } };
							await writer.write(encoder.encode(`data: ${JSON.stringify(errorChunk)}\n\n`));
                            // Always attempt to send DONE, even after error chunk
							await writer.write(encoder.encode(`data: [DONE]\n\n`));
                        } catch (writeError) {
                            console.error(`Error writing final stream error message (Key ${i}):`, writeError);
                        }
                        // Do not re-throw here, as the initial response has already been sent.
                        // The error is reported within the stream itself.
                    } finally {
                        try {
                            console.log(`Closing stream writer (Key ${i})...`);
                            await writer.close();
                            console.log(`Stream writer closed (Key ${i}).`);
                        } catch (closeError) {
                            console.error(`Error closing stream writer (Key ${i}):`, closeError);
                        }
                    }
                })();

                ctx.waitUntil(streamProcessing); // Allow background processing

                // Return the readable stream immediately for the client
                return { readable }; // Indicate success for this key attempt

            } else {
                // --- Non-Stream Handling ---
                console.log(`Executing fal.subscribe with key index ${i}... Payload:`, JSON.stringify(falPayload, null, 2));
                const result = await fal.subscribe("fal-ai/any-llm", falPayload as any) as FalSubscribeResult;
                console.log(`fal.subscribe successful with key index ${i}.`);

                // Log any embedded logs or errors from the non-stream result
                if (result?.data?.logs && Array.isArray(result.data.logs)) {
                    result.data.logs.forEach((log: LogEntry) => console.log(`[Fal Log - Key ${i}] ${log.message}`));
                }
                 if (result?.data?.error) {
                    // Treat error within data as failure for this key
                    console.error(`Fal-ai returned an error in non-stream mode (Key ${i}):`, result.data.error);
                    throw new Error(`Fal error in response data: ${JSON.stringify(result.data.error)}`);
                }

                return result; // Return successful result
            }

        } catch (error: any) {
            lastError = error; // Store the error from this attempt
            console.warn(`Fal request attempt failed for key index ${i}:`, error.message || error);

            // Check if the error suggests a key-specific issue (e.g., auth, rate limit, server error)
            // Fal client might throw custom errors or HTTP status codes might be attached.
            // This check might need refinement based on actual errors thrown by @fal-ai/client.
            // Assuming status property exists for HTTP errors.
            const status = error?.status || error?.response?.status;
            const isRetryableError = status === 401 || status === 403 || status === 429 || (status >= 500 && status < 600) || error.message?.includes('authentication') || error.message?.includes('rate limit');


            if (isRetryableError && (i < falKeys.length - 1)) {
                console.log(`Error is potentially key-related or transient (status: ${status}). Trying next key.`);
                continue; // Try the next key
            } else {
                console.error(`Non-retryable error (status: ${status}) or final key attempt failed for key index ${i}.`);
                throw error; // Re-throw the error if it's not retryable or it's the last key
            }
        }
    }

    // If the loop finishes without returning/throwing successfully, it means all keys failed.
    console.error("All Fal API key attempts failed.");
    // Throw the last recorded error
    throw new Error(`All Fal API key attempts failed. Last error: ${lastError?.message || lastError}`);
}

// === Main Fetch Handler ===
export default {
	async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
		const url = new URL(request.url);
		const path = url.pathname;
		const method = request.method;

		// Handle CORS preflight requests (OPTIONS)
		if (method === 'OPTIONS') {
			 return new Response(null, { 
                status: 204,
                headers: getCorsHeaders()
             });
		}

        let falKeys: string[] = [];

		// --- Authentication & Fal Key Parsing --- (Run for all relevant endpoints)
		if (path.startsWith('/v1/')) { // Only process /v1 routes
			// --- Worker Authentication ---
			const authHeader = request.headers.get('Authorization');
			const apiKeysString = env.API_KEY;

			if (!apiKeysString) {
				console.error("API_KEY secret is not set.");
				return createCorsResponse(
                    JSON.stringify({ error: { message: 'Server configuration error: API Key secret missing.', type: 'server_error' } }), 
                    { status: 500, headers: { 'Content-Type': 'application/json' } }
                );
			}

			const allowedApiKeys = apiKeysString.split(',').map(k => k.trim()).filter(k => k.length > 0);
			if (allowedApiKeys.length === 0) {
				console.error("API_KEY secret is set but contains no valid keys after parsing.");
				return createCorsResponse(
                    JSON.stringify({ error: { message: 'Server configuration error: No valid API Keys found in secret.', type: 'server_error' } }), 
                    { status: 500, headers: { 'Content-Type': 'application/json' } }
                );
			}

			const providedApiKey = authHeader?.startsWith('Bearer ') ? authHeader.substring(7) : null;

			if (!providedApiKey || !allowedApiKeys.includes(providedApiKey)) {
				console.error(`Worker auth failed. Header: ${authHeader ? 'Present' : 'Missing'}, Key check: Provided key "${providedApiKey}" not in allowed list.`);
				return createCorsResponse(
                    JSON.stringify({ error: { message: 'Incorrect API key provided.', type: 'invalid_request_error', param: null, code: 'invalid_api_key'} }), 
                    { status: 401, headers: { 'Content-Type': 'application/json' } }
                );
			}

			// --- Parse Fal Keys ---
			if (!env.FAL_KEY) {
				console.error("FAL_KEY secret is not set.");
				return createCorsResponse(
                    JSON.stringify({ error: { message: 'Server configuration error: Fal Key missing.', type: 'server_error'} }), 
                    { status: 500, headers: { 'Content-Type': 'application/json' } }
                );
			}
			const apikey = env.FAL_KEY1 +env.FAL_KEY2 + env.FAL_KEY3
            falKeys = apikey.split(',').map(k => k.trim()).filter(k => k.length > 0);
            if (falKeys.length === 0) {
                console.error("FAL_KEY secret is set but contains no valid keys after parsing.");
				return createCorsResponse(
                    JSON.stringify({ error: { message: 'Server configuration error: No valid Fal Keys found.', type: 'server_error'} }), 
                    { status: 500, headers: { 'Content-Type': 'application/json' } }
                );
            }
            console.log(`Found ${falKeys.length} Fal API key(s).`);
            // Note: fal.config() is now called inside tryFalRequest for each attempt

		} else if (path === '/') {
			// Allow root path without auth
		} else {
			// Any other path is Not Found
			return createCorsResponse(
                JSON.stringify({ error: { message: `Invalid path: ${path}`, type: 'invalid_request_error'} }), 
                { status: 404, headers: { 'Content-Type': 'application/json' } }
            );
		}

		// --- Routing ---

		// GET /v1/models
		if (method === 'GET' && path === '/v1/models') {
			try {
				const modelsData = FAL_SUPPORTED_MODELS.map(modelId => ({
					id: modelId,
					object: "model",
					created: Math.floor(Date.now() / 1000),
					owned_by: getOwner(modelId)
				}));
				return createCorsResponse(
                    JSON.stringify({ object: "list", data: modelsData }), 
                    { headers: { 'Content-Type': 'application/json' } }
                );
			} catch (error: any) {
				console.error("Error processing GET /v1/models:", error);
				return createCorsResponse(
                    JSON.stringify({ error: { message: `Failed to retrieve model list: ${error.message}`, type: 'server_error'} }), 
                    { status: 500, headers: { 'Content-Type': 'application/json' } }
                );
			}
		}

		// POST /v1/chat/completions
		if (method === 'POST' && path === '/v1/chat/completions') {
			let requestBody: OpenAIChatCompletionRequest;
			try {
				requestBody = await request.json();
			} catch (error) {
				return createCorsResponse(
                    JSON.stringify({ error: { message: 'Invalid JSON request body', type: 'invalid_request_error'} }), 
                    { status: 400, headers: { 'Content-Type': 'application/json' } }
                );
			}

			const { model: requestedModel, messages, stream = false, reasoning_effort } = requestBody;

			if (!requestedModel || !messages || !Array.isArray(messages) || messages.length === 0) {
				return createCorsResponse(
                    JSON.stringify({ error: { message: 'Missing or invalid parameters: model and messages array are required.', type: 'invalid_request_error'} }), 
                    { status: 400, headers: { 'Content-Type': 'application/json' } }
                );
			}

			// Validate and determine the model to use
			let modelToUse: FalModelId;
			if (isValidFalModelId(requestedModel)) {
				modelToUse = requestedModel;
			} else {
				console.error(`Error: Requested model '${requestedModel}' is not supported.`);
				return createCorsResponse(
                    JSON.stringify({
                        error: {
                            message: `The model \`${requestedModel}\` does not exist or is not supported by this endpoint.`,
                            type: 'invalid_request_error',
                            param: 'model',
                            code: 'model_not_found'
                        }
                    }), 
                    { status: 400, headers: { 'Content-Type': 'application/json' } }
                );
			}

			const shouldRequestReasoning = !!reasoning_effort && ['low', 'medium', 'high'].includes(reasoning_effort);

			try {
				const { prompt, system_prompt } = convertMessagesToFalPrompt(messages);
				const falInput: FalInput = {
					model: modelToUse,
					prompt: prompt,
					reasoning: shouldRequestReasoning,
				};
				if (system_prompt) {
					falInput.system_prompt = system_prompt;
				}

                // Prepare the final payload for Fal
                const falPayload: FalPayload = { input: falInput };

				console.log(`Preparing request for fal-ai/any-llm. Model: ${falInput.model}, Stream: ${stream}, Reasoning Effort: ${reasoning_effort ?? 'none'} -> Fal Reasoning: ${falInput.reasoning}`);

                // Use the tryFalRequest function with key rotation
                const result = await tryFalRequest(falKeys, falPayload, stream, requestedModel, ctx);

				if (stream) {
                    // --- Stream Handling ---
                    // tryFalRequest handles the stream setup internally if successful
                    // We just need to return the response with the readable stream
                    if ('readable' in result) {
                        // For streaming responses, we need to add CORS headers but keep the stream
                        const corsHeaders = getCorsHeaders();
                        return new Response(result.readable, {
                            headers: {
                                'Content-Type': 'text/event-stream; charset=utf-8',
                                'Cache-Control': 'no-cache',
                                'Connection': 'keep-alive',
                                ...corsHeaders
                            }
                        });
                    } else {
                         // This case should theoretically not happen if stream=true and no error was thrown
                         console.error("tryFalRequest returned unexpected result for stream=true");
                         throw new Error("Internal server error during stream setup.");
                    }

				} else {
					// --- Non-Stream Handling ---
                    // tryFalRequest returns the FalSubscribeResult directly if successful
					const falResult = result as FalSubscribeResult; // Type assertion

					// Log the raw result for debugging (optional)
					// console.log("Raw non-stream result from fal-ai:", JSON.stringify(falResult, null, 2));

					// Access properties correctly from the data object returned by tryFalRequest
                    const outputContent = falResult?.data?.output ?? "";
                    const reasoningData = falResult?.data?.reasoning;
                    const reqId = falResult?.requestId || Date.now().toString();

					const openAIResponse = {
						id: `chatcmpl-${reqId}`,
						object: "chat.completion",
						created: Math.floor(Date.now() / 1000),
						model: requestedModel, // Report the originally requested model
						choices: [{
							index: 0,
							message: {
								role: "assistant",
								content: outputContent
							},
							finish_reason: "stop",
							logprobs: null,
							...(reasoningData && { reasoning: [{ index: 0, content: reasoningData }] })
						}],
						usage: { prompt_tokens: null, completion_tokens: null, total_tokens: null },
						system_fingerprint: null,
					};
					return createCorsResponse(
                        JSON.stringify(openAIResponse), 
                        { headers: { 'Content-Type': 'application/json' } }
                    );
				}
			} catch (error: any) {
                // Catch errors from tryFalRequest (e.g., all keys failed) or other processing errors
				console.error("Error during Fal request processing or after retries:", error);
                const errorMessage = (error instanceof Error) ? error.message : JSON.stringify(error);
				return createCorsResponse(
                    JSON.stringify({ error: { message: `Error processing request: ${errorMessage}`, type: 'server_error' } }), 
                    { status: 500, headers: { 'Content-Type': 'application/json' } }
                );
			}
		}

		// Default response for root path or other unhandled routes
		return createCorsResponse(
            JSON.stringify({ status: 'ok', message: 'Fal AI-powered OpenAI Proxy API' }), 
            { headers: { 'Content-Type': 'application/json' } }
        );
	}
};