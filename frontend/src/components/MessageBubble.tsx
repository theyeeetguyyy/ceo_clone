import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { ChevronDown, ChevronUp, Zap, FileText } from 'lucide-react';
import { Message } from '../types';

interface Props {
  message: Message;
  onFollowUpClick: (question: string) => void;
}

export default function MessageBubble({ message, onFollowUpClick }: Props) {
  const [showSources, setShowSources] = useState(false);
  const isUser = message.role === 'user';

  if (isUser) {
    return (
      <div className="flex justify-end animate-slide-up">
        <div className="max-w-[80%] bg-blue-600 text-white rounded-2xl rounded-tr-sm px-5 py-3.5 shadow-sm">
          <p className="text-[15px] leading-relaxed">{message.content}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start gap-4 animate-slide-up max-w-[90%]">
      <div className="flex-shrink-0 mt-1 w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center text-white font-bold text-xs shadow-md">
        GA
      </div>
      
      <div className="flex-1 space-y-3 min-w-0">
        {message.isThinking ? (
          <div className="glass-panel rounded-2xl rounded-tl-sm px-5 py-4 w-fit flex items-center gap-3">
            <div className="flex gap-1">
              <div className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-bounce [animation-delay:-0.3s]"></div>
              <div className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-bounce [animation-delay:-0.15s]"></div>
              <div className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-bounce"></div>
            </div>
            <span className="text-sm text-blue-600 font-medium">Govind is thinking...</span>
          </div>
        ) : (
          <div className="glass-panel rounded-2xl rounded-tl-sm px-6 py-5 shadow-lg overflow-hidden">
            <div className="prose prose-slate prose-custom max-w-none text-[15px] text-slate-800">
              <ReactMarkdown>{message.content}</ReactMarkdown>
            </div>
            
            {message.isStreaming && (
              <span className="inline-block w-2 h-4 bg-blue-500 ml-1 animate-pulse align-middle"></span>
            )}
          </div>
        )}

        {/* Sources Accordion */}
        {message.sources && message.sources.length > 0 && !message.isStreaming && (
          <div className="mt-2">
            <button 
              onClick={() => setShowSources(!showSources)}
              className="flex items-center gap-2 text-xs font-medium text-slate-600 hover:text-slate-800 transition-colors bg-slate-100 px-3 py-1.5 rounded-lg border border-slate-200"
            >
              <FileText className="w-3.5 h-3.5" />
              {message.sources.length} Sources Retrieved
              {showSources ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
            </button>
            
            {showSources && (
              <div className="mt-2 grid gap-2">
                {message.sources.map((src, i) => (
                  <div key={i} className="bg-slate-50 border border-slate-200 rounded-lg p-3 text-xs text-slate-600">
                    <div className="flex justify-between items-center mb-1.5">
                      <span className="font-semibold text-blue-600">{src.source}</span>
                      <span className="text-slate-500 bg-slate-200 px-2 py-0.5 rounded text-[10px]">
                        Conf: {(src.score * 100).toFixed(0)}%
                      </span>
                    </div>
                    {src.speaker && <div className="text-slate-500 mb-1">Speaker: {src.speaker} | Date: {src.date}</div>}
                    <div className="text-slate-600 italic border-l-2 border-slate-300 pl-2">"{src.preview}"</div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Follow Up Chips */}
        {message.followUpQuestions && message.followUpQuestions.length > 0 && !message.isStreaming && (
          <div className="flex flex-wrap gap-2 pt-2">
            {message.followUpQuestions.map((q, i) => (
              <button
                key={i}
                onClick={() => onFollowUpClick(q)}
                className="text-xs bg-white hover:bg-blue-50 text-blue-700 border border-slate-200 hover:border-blue-400 rounded-full px-4 py-2 transition-all shadow-sm flex items-center gap-1.5"
              >
                <Zap className="w-3 h-3 text-blue-500" />
                {q}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
