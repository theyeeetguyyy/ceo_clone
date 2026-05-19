export interface Source {
  source: string;
  score: number;
  speaker: string;
  date: string;
  chunk_type: 'fact' | 'style' | 'reasoning';
  preview: string;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  isStreaming?: boolean;
  isThinking?: boolean;
  sources?: Source[];
  confidence?: number;
  routing?: string;
  latency?: number;
  followUpQuestions?: string[];
}
