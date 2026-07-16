import { useEffect, useRef, useState } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const AGENT_LABELS = {
  agent1_tin_registration: "TIN Registration",
  agent2_individual_income_tax: "Individual Income Tax",
  agent3_corporate_income_tax: "Corporate Income Tax",
  agent4_withholding_tax: "Withholding Tax",
  general_fallback: "General / Web Fallback",
};

function agentLabel(key) {
  return AGENT_LABELS[key] || key || "Unknown";
}

function Stamp({ agent }) {
  return (
    <div className="stamp">
      <span className="stamp-dot" />
      Routed &rarr; {agentLabel(agent)}
    </div>
  );
}

function Sources({ sources }) {
  const [open, setOpen] = useState(false);

  if (!sources || sources.length === 0) return null;

  return (
    <div className="sources">
      <button className="sources-toggle" onClick={() => setOpen((o) => !o)}>
        {open ? "Hide" : "Show"} sources ({sources.length})
      </button>
      {open && (
        <ul className="sources-list">
          {sources.map((s, i) => (
            <li key={i}>
              <span className="source-doc">{s.source}</span>
              <p>{s.text}&hellip;</p>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function Message({ msg }) {
  if (msg.role === "user") {
    return (
      <div className="row row-user">
        <div className="bubble bubble-user">{msg.text}</div>
      </div>
    );
  }

  if (msg.role === "error") {
    return (
      <div className="row row-assistant">
        <div className="bubble bubble-error">{msg.text}</div>
      </div>
    );
  }

  return (
    <div className="row row-assistant">
      <div className="bubble bubble-assistant">
        <Stamp agent={msg.agent} />
        <p className="answer-text">{msg.text}</p>
        <Sources sources={msg.sources} />
      </div>
    </div>
  );
}

export default function App() {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      agent: "general_fallback",
      text:
        "Ayubowan! Ask me about TIN registration, individual or corporate income tax, or withholding tax under Sri Lankan law.",
      sources: [],
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, loading]);

  async function sendMessage() {
    const text = input.trim();
    if (!text || loading) return;

    setMessages((m) => [...m, { role: "user", text }]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch(`${API_URL}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `Request failed (${res.status})`);
      }

      const data = await res.json();
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          agent: data.routed_agent,
          text: data.answer,
          sources: data.sources,
        },
      ]);
    } catch (err) {
      setMessages((m) => [
        ...m,
        {
          role: "error",
          text: `Couldn't reach TaxPayBuddy: ${err.message}. Is the API server running on ${API_URL}?`,
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header-inner">
          <h1>TaxPayBuddy</h1>
          <p className="subtitle">Multi-agent assistant for Sri Lankan tax law</p>
        </div>
      </header>

      <main className="chat" ref={scrollRef}>
        <div className="chat-inner">
          {messages.map((msg, i) => (
            <Message key={i} msg={msg} />
          ))}
          {loading && (
            <div className="row row-assistant">
              <div className="bubble bubble-assistant bubble-loading">
                <span className="dot" />
                <span className="dot" />
                <span className="dot" />
              </div>
            </div>
          )}
        </div>
      </main>

      <footer className="composer">
        <div className="composer-inner">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Ask about TIN, income tax, or withholding tax..."
            rows={1}
          />
          <button onClick={sendMessage} disabled={loading || !input.trim()}>
            Send
          </button>
        </div>
      </footer>
    </div>
  );
}
