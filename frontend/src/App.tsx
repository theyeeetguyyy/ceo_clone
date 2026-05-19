import ChatInterface from './components/ChatInterface';

function App() {
  return (
    <div className="min-h-screen bg-slate-50 flex flex-col font-sans">
      <header className="bg-white/80 backdrop-blur-md border-b border-slate-200 py-4 px-6 flex justify-between items-center sticky top-0 z-50">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center text-white font-bold text-lg shadow-lg">
            GA
          </div>
          <div>
            <h1 className="text-xl font-semibold text-slate-900">Govind Agrawal</h1>
            <p className="text-xs text-blue-600 font-medium tracking-wide uppercase">CEO Digital Twin</p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <span className="relative flex h-2.5 w-2.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-500"></span>
            </span>
            <span className="text-xs font-medium text-slate-600">System Online</span>
          </div>
        </div>
      </header>
      
      <main className="flex-1 max-w-4xl w-full mx-auto p-4 md:p-6 flex flex-col h-[calc(100vh-73px)]">
        <ChatInterface />
      </main>
    </div>
  );
}

export default App;
