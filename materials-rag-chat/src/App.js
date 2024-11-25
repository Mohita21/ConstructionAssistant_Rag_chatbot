import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [chatHistory, setChatHistory] = useState([]);
  const [currentSessionId, setCurrentSessionId] = useState(null);
  const messagesEndRef = useRef(null);
  const [isStreaming, setIsStreaming] = useState(false);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const saveCurrentChat = () => {
    if (messages.length > 0 && currentSessionId) {
      setChatHistory(prev => {
        const updatedHistory = prev.map(chat => 
          chat.id === currentSessionId 
            ? { ...chat, messages: [...messages] }
            : chat
        );
        
        if (!prev.find(chat => chat.id === currentSessionId)) {
          updatedHistory.push({
            id: currentSessionId,
            title: messages[0].content.substring(0, 30) + "...",
            thumbnail: getChatThumbnail(messages[0].content),
            messages: [...messages]
          });
        }
        
        return updatedHistory;
      });
    }
  };

  const createNewSession = async () => {
    try {
      saveCurrentChat();

      setMessages([]);
      const newSessionId = Date.now();
      setCurrentSessionId(newSessionId);

      const sidebarItems = document.querySelectorAll('.chat-history-item');
      sidebarItems.forEach(item => item.classList.remove('active'));
      
      return newSessionId;
    } catch (error) {
      console.error('Error creating new session:', error);
    }
  };

  const loadChat = (chatSession) => {
    saveCurrentChat();
    
    setCurrentSessionId(chatSession.id);
    setMessages(chatSession.messages);

    const sidebarItems = document.querySelectorAll('.chat-history-item');
    sidebarItems.forEach(item => item.classList.remove('active'));
    document.querySelector(`[data-id="${chatSession.id}"]`)?.classList.add('active');
  };

  const getChatClassName = (chatId) => {
    return `chat-history-item ${chatId === currentSessionId ? 'active' : ''}`;
  };

  const abortController = new AbortController();

  const getChatThumbnail = (content) => {
    const lowerContent = content.toLowerCase();
    
    if (lowerContent.includes('concrete') || lowerContent.includes('cement')) {
      return 'fa-cubes';
    } else if (lowerContent.includes('wood') || lowerContent.includes('timber')) {
      return 'fa-tree';
    } else if (lowerContent.includes('steel') || lowerContent.includes('metal')) {
      return 'fa-industry';
    } else if (lowerContent.includes('cost') || lowerContent.includes('price')) {
      return 'fa-dollar-sign';
    } else if (lowerContent.includes('safety') || lowerContent.includes('protection')) {
      return 'fa-hard-hat';
    } else if (lowerContent.includes('plan') || lowerContent.includes('design')) {
      return 'fa-drafting-compass';
    }
    return 'fa-building';
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);
    setIsStreaming(true);

    if (messages.length === 0) {
      const thumbnail = getChatThumbnail(input);
      const chatTitle = input.substring(0, 30) + "...";
      setChatHistory(prev => [...prev, {
        id: currentSessionId,
        title: chatTitle,
        thumbnail: thumbnail,
        messages: [userMessage]
      }]);
    }

    try {
      const response = await fetch('http://localhost:8000/query/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: input,
          init_new_session: messages.length === 0,
          session_id: currentSessionId
        }),
        signal: abortController.signal
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      setMessages(prev => [...prev, {
        role: 'assistant',
        content: '',
        sources: []
      }]);

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (!line.trim() || !line.startsWith('data:')) continue;
          
          try {
            const data = JSON.parse(line.replace('data:', ''));
            
            setMessages(prev => {
              const newMessages = [...prev];
              const currentMessage = newMessages[newMessages.length - 1];

              switch (data.type) {
                case 'token':
                  const newContent = data.content.replace(/\s+/g, ' ');  // Replace multiple spaces with single space
                  if (!currentMessage.content.includes(newContent)) {
                    currentMessage.content += newContent;
                  }
                  break;
                case 'source':
                  currentMessage.sources = data.content;
                  break;
                case 'error':
                  currentMessage.content = 'Error: ' + data.content;
                  break;
              }

              return newMessages;
            });
          } catch (error) {
            console.error('Error parsing chunk:', error);
          }
        }
      }
    } catch (error) {
      if (error.name === 'AbortError') {
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: 'Request cancelled.'
        }]);
      } else {
        console.error('Error sending message:', error);
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: 'Sorry, there was an error processing your request.'
        }]);
      }
    } finally {
      setIsLoading(false);
      setIsStreaming(false);
    }
  };

  const deleteChat = (chatId, e) => {
    e.stopPropagation();
    setChatHistory(prev => prev.filter(chat => chat.id !== chatId));
  };

  const cleanResponse = (text) => {
    // Remove XML tags and unwanted characters
    let cleaned = text.replace(/<answer>|<\/answer>/g, '').trim();
    cleaned = cleaned.replace(/[#*]+/g, ''); // Remove # and * characters
    
    // Fix O.C. formatting
    cleaned = cleaned.replace(/\(O\.\s*C\.\)/g, '(O.C.)');
    cleaned = cleaned.replace(/O\.\s*C\./g, 'O.C.');
    
    // Format sections with proper spacing
    const sections = {
      'Relevant Information': '',
      'Joist Spacing': '',
      'Material Preparation': '',
      'General Recommendations': '',
      'Conclusion': '',
      'Best Practices': ''
    };

    // Add bold formatting to section headers
    Object.keys(sections).forEach(section => {
      cleaned = cleaned.replace(
        new RegExp(`${section}:?`, 'g'), 
        `\n\n**${section}:**\n`
      );
    });

    // Format list items
    cleaned = cleaned.replace(/^-\s*/gm, '• '); // Convert dashes to bullets
    cleaned = cleaned.replace(/For span following/g, '\nFor span following');
    
    // Fix spacing and line breaks
    cleaned = cleaned
      .replace(/\s+/g, ' ') // Normalize spaces
      .replace(/\n\s+/g, '\n') // Remove spaces after newlines
      .replace(/:\s+/g, ': ') // Fix spacing after colons
      .replace(/\n{3,}/g, '\n\n') // Replace multiple newlines with double
      .replace(/\s*\n\s*/g, '\n') // Clean up spaces around newlines
      .trim();

    // Add proper line breaks for readability
    cleaned = cleaned
      .split('\n')
      .map(line => line.trim())
      .filter(line => line) // Remove empty lines
      .join('\n');

    // Format lumber dimensions consistently
    cleaned = cleaned.replace(/(\d+)x(\d+)/g, '$1x$2');
    
    return cleaned;
  };

  return (
    <div className="app">
      <div className="sidebar">
        <button onClick={createNewSession} className="new-chat-btn">
          New Chat
        </button>
        <div className="chat-history">
          {chatHistory.map((chat) => (
            <div 
              key={chat.id}
              data-id={chat.id}
              className={getChatClassName(chat.id)}
              onClick={() => loadChat(chat)}
            >
              <div className="chat-item-content">
                <i className={`fas ${chat.thumbnail || 'fa-building'} chat-thumbnail`}></i>
                <span className="chat-title">
                  {chat.title || "New Chat"}
                </span>
              </div>
              <button 
                className="delete-chat-btn"
                onClick={(e) => deleteChat(chat.id, e)}
              >
                ×
              </button>
            </div>
          ))}
        </div>
      </div>
      <div className="main">
        <div className="chat-title-header">
          <h1>
            <i className="fas fa-hard-hat"></i>
            <i className="fas fa-home"></i>
            Construction Assistant
            <i className="fas fa-tools"></i>
            <i className="fas fa-hammer"></i>
          </h1>
          <p className="tagline">We help building your dreams...</p>
        </div>
        <div className="chat-container">
          {messages.map((message, index) => (
            <div key={index} className={`message ${message.role}`}>
              <div className="message-content">
                {message.role === 'user' ? (
                  <p>{message.content}</p>
                ) : (
                  <ReactMarkdown>{cleanResponse(message.content)}</ReactMarkdown>
                )}
              </div>
              {message.role === 'assistant' && message.sources && message.sources.length > 0 && (
                <div className="sources">
                  <h4>Sources:</h4>
                  <ul>
                    <li>
                      <ReactMarkdown>{message.sources[0]}</ReactMarkdown>
                    </li>
                  </ul>
                </div>
              )}
            </div>
          ))}
          {isLoading && <div className="loading">Thinking...</div>}
          <div ref={messagesEndRef} />
        </div>
        <form onSubmit={handleSubmit} className="input-form">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="How can I help you?"
            disabled={isLoading}
          />
          <button type="submit" disabled={isLoading}>
            Send
          </button>
          <button onClick={() => abortController.abort()}>
            Cancel
          </button>
        </form>
      </div>
    </div>
  );
}

export default App;