import { useState, useRef, useEffect } from 'react';
import { Send, Mic, Square, Trash2, VolumeX } from 'lucide-react';
import { usePostHog } from 'posthog-js/react';
import { useSSEStream } from '../hooks/useSSEStream';
import { useVoice } from '../hooks/useVoice';
import MessageBubble from './MessageBubble';

export default function ChatInterface() {
  const [input, setInput] = useState('');
  const [isVoiceMode, setIsVoiceMode] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  
  const { messages, isGenerating, sendMessage, stopGeneration, clearHistory } = useSSEStream();
  const posthog = usePostHog();
  
  // Voice integration
  const { isRecording, toggleRecording, speak, stopSpeaking } = useVoice((text, voiceMode) => {
    setIsVoiceMode(voiceMode);
    sendMessage(text, voiceMode ? 'voice' : 'text');
    posthog?.capture('Asked Question', { question: text, mode: voiceMode ? 'voice' : 'text' });
  });

  // Speak completed messages if in voice mode
  useEffect(() => {
    if (isVoiceMode && messages.length > 0) {
      const lastMsg = messages[messages.length - 1];
      if (lastMsg.role === 'assistant' && !lastMsg.isStreaming && !lastMsg.isThinking) {
        speak(lastMsg.content);
        // Reset voice mode flag after speaking so text messages don't get spoken
        setIsVoiceMode(false); 
      }
    }
  }, [messages, isVoiceMode, speak]);

  // Auto-scroll
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isGenerating]);

  const handleSubmit = (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!input.trim() || isGenerating) return;
    sendMessage(input, 'text');
    posthog?.capture('Asked Question', { question: input, mode: 'text' });
    setInput('');
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="flex flex-col h-full bg-white/60 rounded-2xl border border-slate-200 shadow-xl overflow-hidden relative">
      {/* Chat History */}
      <div className="flex-1 overflow-y-auto p-4 md:p-6 space-y-6">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-center opacity-80 animate-fade-in">
            <div className="w-20 h-20 bg-blue-50 rounded-full flex items-center justify-center mb-6 ring-1 ring-blue-100">
              <Mic className="w-8 h-8 text-blue-500" />
            </div>
            <h2 className="text-2xl font-semibold text-slate-800 mb-2">Welcome to Anaxee</h2>
            <p className="text-slate-500 max-w-md">
              You are speaking with the digital twin of Govind Agrawal. 
              Ask about Anaxee's vision, strategy in tier 2/3 cities, or recent updates.
            </p>
          </div>
        ) : (
          messages.map(msg => (
            <MessageBubble key={msg.id} message={msg} onFollowUpClick={(q) => sendMessage(q, 'text')} />
          ))
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input Area */}
      <div className="p-4 bg-white/80 backdrop-blur-lg border-t border-slate-200">
        <div className="flex items-center justify-between mb-3 px-2">
          <button 
            onClick={clearHistory}
            className="text-xs text-slate-500 hover:text-slate-700 flex items-center gap-1 transition-colors"
          >
            <Trash2 className="w-3 h-3" /> Clear Chat
          </button>
          
          {isVoiceMode && (
            <button 
              onClick={() => { stopSpeaking(); setIsVoiceMode(false); }}
              className="text-xs text-blue-400 hover:text-blue-300 flex items-center gap-1 transition-colors animate-pulse-slow"
            >
              <VolumeX className="w-3 h-3" /> Stop Speaking
            </button>
          )}
        </div>

        <form onSubmit={handleSubmit} className="relative flex items-end gap-2">
          <div className="relative flex-1 bg-slate-100 rounded-xl border border-slate-300 focus-within:border-blue-500 focus-within:ring-1 focus-within:ring-blue-500/20 transition-all overflow-hidden">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Message Govind..."
              className="w-full max-h-32 min-h-[56px] bg-transparent text-slate-800 placeholder-slate-400 p-4 resize-none focus:outline-none"
              rows={1}
            />
          </div>
          
          <div className="flex gap-2 h-[56px]">
            <button
              type="button"
              onClick={toggleRecording}
              className={`flex items-center justify-center w-14 rounded-xl transition-all ${
                isRecording 
                  ? 'bg-red-50 text-red-500 border border-red-200 animate-pulse' 
                  : 'glass-button text-slate-600'
              }`}
            >
              {isRecording ? <Square className="w-5 h-5 fill-current" /> : <Mic className="w-5 h-5" />}
            </button>

            {isGenerating ? (
              <button
                type="button"
                onClick={stopGeneration}
                className="flex items-center justify-center w-14 rounded-xl glass-button text-red-400"
              >
                <Square className="w-5 h-5 fill-current" />
              </button>
            ) : (
              <button
                type="submit"
                disabled={!input.trim()}
                className="flex items-center justify-center w-14 rounded-xl bg-blue-600 hover:bg-blue-500 text-white disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                <Send className="w-5 h-5" />
              </button>
            )}
          </div>
        </form>
      </div>
    </div>
  );
}
