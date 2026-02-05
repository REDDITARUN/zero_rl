#!/usr/bin/env node

import process from "node:process";

function emit(payload) {
  return new Promise((resolve) => {
    const line = `${JSON.stringify(payload)}\n`;
    const ok = process.stdout.write(line);
    if (ok) {
      resolve();
      return;
    }
    process.stdout.once("drain", () => resolve());
  });
}

function readStdin() {
  return new Promise((resolve, reject) => {
    let data = "";
    process.stdin.setEncoding("utf8");
    process.stdin.on("data", (chunk) => {
      data += chunk;
    });
    process.stdin.on("end", () => resolve(data));
    process.stdin.on("error", reject);
  });
}

function summarizeItem(item, eventType) {
  if (!item || typeof item !== "object") {
    return "";
  }

  const kind = item.type;
  if (kind === "reasoning" && typeof item.text === "string") {
    return `Reasoning: ${item.text.slice(0, 200)}`;
  }
  if (kind === "agent_message" && typeof item.text === "string") {
    return eventType === "item.completed" ? "Response prepared" : "Drafting response";
  }
  if (kind === "command_execution" && typeof item.command === "string") {
    const status = item.status || "running";
    return `Command ${status}: ${item.command.slice(0, 120)}`;
  }
  if (kind === "mcp_tool_call" && typeof item.tool === "string") {
    const status = item.status || "running";
    return `Tool ${status}: ${item.server}/${item.tool}`;
  }
  if (kind === "web_search" && typeof item.query === "string") {
    return `Searching: ${item.query.slice(0, 120)}`;
  }
  if (kind === "todo_list" && Array.isArray(item.items)) {
    const done = item.items.filter((x) => x && x.completed).length;
    return `Plan: ${done}/${item.items.length} done`;
  }
  if (kind === "error" && typeof item.message === "string") {
    return `Item error: ${item.message}`;
  }
  return "";
}

async function main() {
  try {
    const raw = await readStdin();
    const payload = JSON.parse(raw);

    const {
      agent_id,
      prompt,
      model,
      output_schema,
      working_directory,
      sandbox_mode,
      approval_policy,
      network_access_enabled,
      api_key,
      thread_id,
    } = payload;

    if (!api_key) {
      throw new Error("Missing API key. Set OPENAI_API_KEY or CODEX_API_KEY.");
    }

    const { Codex } = await import("@openai/codex-sdk");

    const codex = new Codex({
      apiKey: api_key,
      baseUrl: process.env.OPENAI_BASE_URL || undefined,
      env: {
        ...process.env,
        CODEX_API_KEY: api_key,
      },
      config: {
        show_raw_agent_reasoning: false,
      },
    });

    const thread = thread_id
      ? codex.resumeThread(thread_id)
      : codex.startThread({
          model: model || process.env.CODEX_MODEL || "gpt-5-codex",
          workingDirectory: working_directory || process.cwd(),
          skipGitRepoCheck: true,
          sandboxMode: sandbox_mode || "read-only",
          approvalPolicy: approval_policy || "never",
          networkAccessEnabled: Boolean(network_access_enabled),
        });

    let finalResponse = "";
    let usage = null;
    let activeThreadId = thread_id || thread.id || null;
    try {
      const streamed = await thread.runStreamed(prompt, { outputSchema: output_schema });
      for await (const event of streamed.events) {
        if (event.type === "thread.started") {
          activeThreadId = event.thread_id;
          await emit({
            type: "progress",
            agent_id,
            thread_id: activeThreadId,
            status: "working",
            message: "Thread started",
          });
        }

        if (event.type === "item.started" || event.type === "item.updated" || event.type === "item.completed") {
          const message = summarizeItem(event.item, event.type);
          if (message) {
            await emit({
              type: "progress",
              agent_id,
              thread_id: activeThreadId,
              status: "working",
              message,
            });
          }
        }

        if (event.type === "item.completed") {
          if (event.item.type === "agent_message" && typeof event.item.text === "string") {
            finalResponse = event.item.text;
          }
        }
        if (event.type === "turn.completed") {
          usage = event.usage;
        }
        if (event.type === "turn.failed") {
          throw new Error(event.error?.message || "Codex turn failed");
        }
        if (event.type === "error") {
          throw new Error(event.message || "Codex stream error");
        }
      }
    } catch (streamErr) {
      await emit({
        type: "progress",
        agent_id,
        thread_id: activeThreadId,
        status: "working",
        message: "Stream interrupted, retrying turn",
      });
      const turn = await thread.run(prompt, { outputSchema: output_schema });
      finalResponse = typeof turn.finalResponse === "string" ? turn.finalResponse : "";
      usage = turn.usage || null;
    }

    if (!finalResponse) {
      await emit({
        type: "progress",
        agent_id,
        thread_id: activeThreadId,
        status: "working",
        message: "No streamed response, retrying buffered turn",
      });
      const turn = await thread.run(prompt, { outputSchema: output_schema });
      finalResponse = typeof turn.finalResponse === "string" ? turn.finalResponse : "";
      usage = turn.usage || usage;
    }

    if (!finalResponse) {
      throw new Error("Codex returned empty final response");
    }

    await emit({
      type: "result",
      ok: true,
      agent_id,
      thread_id: activeThreadId || thread.id || thread_id || null,
      usage,
      final_response: finalResponse,
    });
    process.exitCode = 0;
    return;
  } catch (error) {
    await emit(
      {
        type: "result",
        ok: false,
        error: error instanceof Error ? error.message : String(error),
      },
    );
    process.exitCode = 1;
    return;
  }
}

await main();
