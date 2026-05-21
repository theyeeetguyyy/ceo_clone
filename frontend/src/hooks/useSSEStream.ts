import { useState, useCallback, useRef } from 'react';
import { Message } from '../types';

export function useSSEStream() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const sendMessage = useCallback(async (text: string, mode: 'text' | 'voice' = 'text') => {
    if (!text.trim()) return;

    // Add user message
    const userMsg: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: text,
      timestamp: new Date()
    };
    
    setMessages(prev => [...prev, userMsg]);
    setIsGenerating(true);

    // Create placeholder for assistant response
    const assistantId = (Date.now() + 1).toString();
    setMessages(prev => [...prev, {
      id: assistantId,
      role: 'assistant',
      content: '',
      timestamp: new Date(),
      isStreaming: true,
      isThinking: true
    }]);

    abortControllerRef.current = new AbortController();

    try {
      const response = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: text,
          session_id: sessionId,
          mode: mode
        }),
        signal: abortControllerRef.current.signal
      });

      if (!response.body) throw new Error('No response body');

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let assistantContent = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ') && line !== 'data: [DONE]') {
            try {
              const data = JSON.parse(line.slice(6));
              
              if (data.type === 'session' && !sessionId) {
                setSessionId(data.session_id);
              } 
              else if (data.type === 'thinking') {
                // UI shows thinking indicator natively via isThinking flag
              }
              else if (data.type === 'token') {
                assistantContent += data.content;
                setMessages(prev => prev.map(m => 
                  m.id === assistantId ? { ...m, content: assistantContent, isThinking: false } : m
                ));
              }
              else if (data.type === 'done') {
                setMessages(prev => prev.map(m => 
                  m.id === assistantId ? { 
                    ...m, 
                    content: data.answer,
                    isStreaming: false,
                    isThinking: false,
                    sources: data.sources,
                    confidence: data.confidence,
                    routing: data.routing,
                    latency: data.latency,
                    followUpQuestions: data.follow_up_questions
                  } : m
                ));
              }
            } catch (e) {
              console.error("Error parsing SSE JSON:", e, line);
            }
          }
        }
      }
    } catch (err: any) {
      if (err.name === 'AbortError') {
        console.log('Stream aborted');
      } else {
        console.error('Chat error:', err);
        setMessages(prev => prev.map(m => 
          m.id === assistantId ? { ...m, content: 'Error: Connection failed.', isStreaming: false, isThinking: false } : m
        ));
      }
    } finally {
      setIsGenerating(false);
      abortControllerRef.current = null;
    }
  }, [sessionId]);

  const stopGeneration = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
  }, []);

  const clearHistory = useCallback(async () => {
    if (sessionId) {
      await fetch(`/api/chat/history/${sessionId}`, { method: 'DELETE' });
      setSessionId(null);
    }
    setMessages([]);
  }, [sessionId]);

  return { messages, isGenerating, sendMessage, stopGeneration, clearHistory };
}
