document.addEventListener('DOMContentLoaded', function() {
    const chatForm = document.getElementById('chatForm');
    const userInput = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');
    const chatMessages = document.getElementById('chatMessages');
    const suggestedPrompts = document.getElementById('suggestedPrompts');
    const promptButtons = document.querySelectorAll('.prompt-btn');

    // Auto-expand textarea as user types
    userInput.addEventListener('input', function() {
        // Enable/disable send button based on content
        sendButton.disabled = !userInput.value.trim();
        
        // Hide suggested prompts when user starts typing
        if (userInput.value.trim()) {
            const suggestedPrompts = document.getElementById('suggestedPrompts');
            if (suggestedPrompts) {
                suggestedPrompts.style.display = 'none';
            }
        }
        
        // Auto-expand textarea (reset height first to get accurate scrollHeight)
        this.style.height = 'auto';
        const newHeight = Math.min(this.scrollHeight, 120); // Cap at 120px
        this.style.height = newHeight + 'px';
    });

    // Handle Enter key to send message (with Shift+Enter for new line)
    userInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            const message = userInput.value.trim();
            if (message && !sendButton.disabled) {
                sendMessage(message);
                userInput.value = '';
                sendButton.disabled = true;
                userInput.style.height = 'auto';
            }
        }
    });

    // Handle form submission
    chatForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const message = userInput.value.trim();
        if (message) {
            sendMessage(message);
            userInput.value = '';
            sendButton.disabled = true;
            userInput.style.height = 'auto';
        }
    });
    
    // Handle suggestion prompt buttons
    promptButtons.forEach(button => {
        button.addEventListener('click', function() {
            const promptText = this.getAttribute('data-prompt');
            userInput.value = promptText;
            sendButton.disabled = false;
            // Focus on input so user can modify if desired
            userInput.focus();
            
            // Auto-send after a short delay if user doesn't modify
            setTimeout(() => {
                if (userInput.value === promptText) {
                    sendMessage(promptText);
                    userInput.value = '';
                    sendButton.disabled = true;
                }
            }, 800);
        });
    });

    // Initialize chat with a welcome message (without feedback options)
    addBotWelcomeMessage("Hi! 👋 How can I help you today?");

    // Functions for chat interaction
    function addUserMessage(message) {
        const messageContainer = document.createElement('div');
        messageContainer.className = 'message-container';
        messageContainer.style.alignSelf = 'flex-end';
        messageContainer.setAttribute('role', 'listitem');
        messageContainer.setAttribute('aria-label', 'Your message');

        const messageEl = document.createElement('div');
        messageEl.className = 'message user-message';
        messageEl.textContent = message;

        messageContainer.appendChild(messageEl);
        chatMessages.appendChild(messageContainer);
        scrollToBottom();
    }

    function addBotMessage(message, citations = null, citation_objects = null, graphics = null) {
        const messageContainer = document.createElement('div');
        messageContainer.className = 'message-container';
        messageContainer.setAttribute('role', 'listitem');
        messageContainer.setAttribute('aria-label', 'Abby\'s response');

        const messageEl = document.createElement('div');
        messageEl.className = 'message bot-message';
        
        // PROCESS CITATION OBJECTS FIRST
        let processedCitations = [];
        let citationMap = {};
        
        // Debug - See what citation_objects we received
        console.log("Raw citation objects:", JSON.stringify(citation_objects));
        
        // Process and clean up citation data if available
        if (citation_objects && citation_objects.length > 0) {
            // Check if we have proper objects or just strings
            const isObjectFormat = typeof citation_objects[0] === 'object';
            
            // Debug logging for citation objects
            console.log("Citation objects received:", JSON.stringify(citation_objects));
            
            // First pass - Create citation mapping and processed citations
            citation_objects.forEach((citation, index) => {
                // Use 1-based indexing for citations
                const refNumber = index + 1;
                
                // Handle when citation objects are actual objects with URL properties
                if (isObjectFormat) {
                    // Skip citations without valid source or title
                    if (!citation.source && !citation.title) {
                        return;
                    }
                    
                    // Get URL (optional)
                    const url = citation.url || null;
                    console.log(`Processing citation ${refNumber}: ${citation.source || citation.title}, URL: ${url}`);
                    
                    // Create citation mapping with correct reference number
                    citationMap[refNumber] = {
                        ...citation,
                        referenceNumber: refNumber,
                        url: url
                    };
                    
                    // Determine display name - prefer title, then source
                    let displayName = citation.title || citation.source;
                    
                    // Special case for Planned Parenthood - use consistent naming
                    if (url && url.includes('plannedparenthood.org') && !displayName.includes('Planned Parenthood')) {
                        displayName = 'Planned Parenthood: ' + displayName;
                    }
                    
                    // Add to processed citations
                    processedCitations.push({
                        referenceNumber: refNumber,
                        displayName: displayName,
                        url: url
                    });
                } 
                // Handle when citation_objects are just string names without URLs
                else {
                    const sourceName = citation;
                    
                    // Add to citation map
                    citationMap[refNumber] = {
                        source: sourceName,
                        title: sourceName,
                        referenceNumber: refNumber,
                        url: null
                    };
                    
                    // Add to processed citations
                    processedCitations.push({
                        referenceNumber: refNumber,
                        displayName: sourceName,
                        url: null
                    });
                }
            });
        } else if (citations && citations.length > 0) {
            // If we only have citation strings but no objects
            const uniqueCitations = [...new Set(citations)].filter(c => 
                c && c !== 'Source' && c.length > 0
            );
            
            uniqueCitations.forEach((citation, index) => {
                const refNumber = index + 1;
                
                // Try to extract URL if included
                const urlMatch = citation.match(/(https?:\/\/[^\s]+)/);
                const url = urlMatch ? urlMatch[1] : null;
                
                // Set display name, removing URL if present
                let displayName = citation;
                if (url) {
                    displayName = citation.replace(url, '').trim();
                    displayName = displayName.replace(/:\s*$/, '');
                }
                
                // Add to citation map
                citationMap[refNumber] = {
                    source: displayName || citation,
                    title: displayName || citation,
                    referenceNumber: refNumber,
                    url: url
                };
                
                // Add to processed citations
                processedCitations.push({
                    referenceNumber: refNumber,
                    displayName: displayName || citation,
                    url: url
                });
            });
        }
        
        // Debug - Show citation map
        console.log("Citation map after processing:", JSON.stringify(citationMap));
        
        // PROCESS MESSAGE TEXT
        let formattedMessage = message;
        
        // Clean up any inline citations in text
        
        // 1. Remove citations in parentheses like (Planned Parenthood, SOURCE...)
        formattedMessage = formattedMessage.replace(/\s?\([^)]*(?:SOURCE|source)[^)]*\)/g, '');
        
        // 2. Remove "SOURCE" text
        formattedMessage = formattedMessage.replace(/\s?SOURCE.+?(?=\s|$|\.|,)/g, '');
        
        // 3. Process existing numbered citations [1], [2], etc.
        formattedMessage = formattedMessage.replace(/\[(\d+)\]/g, (match, numStr) => {
            const num = parseInt(numStr);
            
            // Create set to track which citations are already in the text
            if (!window._existingCitations) {
                window._existingCitations = new Set();
            }
            window._existingCitations.add(num);
            
            return `[${num}]`;
        });
        
        // 4. Remove any reference to "For more information, see sources:"
        formattedMessage = formattedMessage.replace(/For more (?:detailed )?information,?\s*(?:you can )?(?:refer to|see|check) (?:the )?(?:resources|sources)(?:\s*from [^.]+)?\.?\s*$/i, '');
        
        // Enhanced markdown conversion
        formattedMessage = formattedMessage
            // Bold text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            // Italic text
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            // Code blocks
            .replace(/```(.*?)```/gs, '<pre><code>$1</code></pre>')
            // Inline code
            .replace(/`(.*?)`/g, '<code>$1</code>')
            // Bullet points
            .replace(/^\s*-\s+(.*?)$/gm, '<li>$1</li>')
            // Numbered lists
            .replace(/^\s*(\d+)\.\s+(.*?)$/gm, '<li>$2</li>')
            // Headers
            .replace(/^###\s+(.*?)$/gm, '<h5>$1</h5>')
            .replace(/^##\s+(.*?)$/gm, '<h4>$1</h4>')
            .replace(/^#\s+(.*?)$/gm, '<h3>$1</h3>');
        
        // Replace line breaks and wrap lists
        formattedMessage = formattedMessage
            .replace(/<\/li>\n<li>/g, '</li><li>') // Combine list items
            .replace(/(<li>.*?<\/li>)/gs, '<ul>$1</ul>') // Wrap lists
            .replace(/<\/ul>\n<ul>/g, '') // Combine adjacent lists
            .replace(/\n\n/g, '<br><br>')
            .replace(/\n/g, '<br>');
        
        // Log citationMap for debugging
        console.log("Citation map before replacing in text:", JSON.stringify(citationMap));
        
        // Replace citation references with links
        formattedMessage = formattedMessage.replace(/\[(\d+)\]/g, (match, citationNumber) => {
            const num = parseInt(citationNumber);
            console.log(`Processing citation reference [${num}] in text`);
            
            // Get the citation from our map
            const citation = citationMap[num];
            if (citation) {
                console.log(`Found citation ${num} in map:`, JSON.stringify(citation));
                
                // If citation has URL, use it
                if (citation.url) {
                    console.log(`Using URL for citation ${num}: ${citation.url}`);
                    return `<sup class="citation-reference">[<a href="${citation.url}" target="_blank" rel="noopener noreferrer">${num}</a>]</sup>`;
                } else {
                    console.log(`No URL for citation ${num}, linking to footnote`);
                    return `<sup class="citation-reference">[<a href="#citation-${num}">${num}</a>]</sup>`;
                }
            } else {
                console.log(`Citation ${num} not found in map, using fallback`);
                return `<sup class="citation-reference">[${num}]</sup>`;
            }
        });
        
        messageEl.innerHTML = formattedMessage;
        messageContainer.appendChild(messageEl);

        // Add message ID as a data attribute for feedback
        const messageId = 'msg_' + new Date().getTime();
        messageEl.dataset.messageId = messageId;

        // ADD CITATIONS SECTION (if we have valid sources)
        if (processedCitations.length > 0) {
            // Sort citations by reference number
            processedCitations.sort((a, b) => a.referenceNumber - b.referenceNumber);
            
            const citationsContainer = document.createElement('div');
            citationsContainer.className = 'citations-container';
            citationsContainer.setAttribute('role', 'contentinfo');
            citationsContainer.setAttribute('aria-label', 'Information sources');
            
            const citationsTitle = document.createElement('div');
            citationsTitle.className = 'citations-title';
            citationsTitle.textContent = 'Sources:';
            
            const citationsList = document.createElement('div');
            citationsList.className = 'citations-list';
            
            // Log processed citations before display
            console.log("Processed citations for display:", JSON.stringify(processedCitations));
            
            // Create citation items
            processedCitations.forEach(citation => {
                const citationItem = document.createElement('div');
                citationItem.className = 'citation';
                citationItem.id = `citation-${citation.referenceNumber}`;
                
                console.log(`Rendering citation ${citation.referenceNumber}: ${citation.displayName}, URL: ${citation.url}`);
                
                // Always use the specific URL if available
                if (citation.url) {
                    let displayName = citation.displayName;
                    citationItem.innerHTML = `<span class="citation-number">[${citation.referenceNumber}]</span> <a href="${citation.url}" target="_blank" rel="noopener noreferrer">${displayName}</a>`;
                } else if (citation.displayName.toLowerCase().includes('planned parenthood')) {
                    // If it's Planned Parenthood without URL, add a generic link
                    citationItem.innerHTML = `<span class="citation-number">[${citation.referenceNumber}]</span> <a href="https://www.plannedparenthood.org" target="_blank" rel="noopener noreferrer">${citation.displayName}</a>`;
                } else {
                    // No URL provided
                    citationItem.innerHTML = `<span class="citation-number">[${citation.referenceNumber}]</span> ${citation.displayName}`;
                }
                
                citationsList.appendChild(citationItem);
            });
            
            citationsContainer.appendChild(citationsTitle);
            citationsContainer.appendChild(citationsList);
            messageContainer.appendChild(citationsContainer);
        }

        // Add graphics if provided
        if (graphics && graphics.length > 0) {
            const graphicsContainer = document.createElement('div');
            graphicsContainer.className = 'graphics-container';
            graphicsContainer.setAttribute('role', 'figure');
            
            graphics.forEach(graphic => {
                const graphicItem = document.createElement('div');
                graphicItem.className = 'graphic';
                
                const graphicTitle = document.createElement('h4');
                graphicTitle.textContent = graphic.title;
                
                const graphicDesc = document.createElement('div');
                graphicDesc.className = 'graphic-description';
                graphicDesc.textContent = graphic.description;
                
                const svgContainer = document.createElement('div');
                svgContainer.className = 'svg-container';
                svgContainer.innerHTML = graphic.svg;
                svgContainer.setAttribute('aria-label', graphic.title);
                
                graphicItem.appendChild(graphicTitle);
                graphicItem.appendChild(graphicDesc);
                graphicItem.appendChild(svgContainer);
                
                graphicsContainer.appendChild(graphicItem);
            });
            
            messageContainer.appendChild(graphicsContainer);
        }

        // Add feedback options
        const feedbackContainer = document.createElement('div');
        feedbackContainer.className = 'feedback-container';
        
        const feedbackLabel = document.createElement('span');
        feedbackLabel.className = 'feedback-label';
        feedbackLabel.textContent = 'Was this helpful?';
        
        const thumbsUp = document.createElement('button');
        thumbsUp.className = 'feedback-btn thumbs-up';
        thumbsUp.innerHTML = '<i class="fas fa-thumbs-up"></i>';
        thumbsUp.setAttribute('aria-label', 'This was helpful');
        thumbsUp.addEventListener('click', function() {
            submitFeedback(messageId, 1);
            feedbackContainer.innerHTML = '<span class="feedback-thanks">Thanks for your feedback!</span>';
        });
        
        const thumbsDown = document.createElement('button');
        thumbsDown.className = 'feedback-btn thumbs-down';
        thumbsDown.innerHTML = '<i class="fas fa-thumbs-down"></i>';
        thumbsDown.setAttribute('aria-label', 'This was not helpful');
        thumbsDown.addEventListener('click', function() {
            submitFeedback(messageId, -1);
            feedbackContainer.innerHTML = '<span class="feedback-thanks">Thanks for your feedback!</span>';
        });
        
        feedbackContainer.appendChild(feedbackLabel);
        feedbackContainer.appendChild(thumbsUp);
        feedbackContainer.appendChild(thumbsDown);
        
        messageContainer.appendChild(feedbackContainer);
        
        chatMessages.appendChild(messageContainer);
        scrollToBottom();
    }

    function addTypingIndicator() {
        const typingIndicator = document.createElement('div');
        typingIndicator.className = 'typing-indicator';
        typingIndicator.id = 'typingIndicator';
        
        // Create dots container
        const dotsContainer = document.createElement('div');
        dotsContainer.className = 'typing-dots';
        
        // Create three dots
        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('div');
            dot.className = 'dot';
            dotsContainer.appendChild(dot);
        }
        
        // Add dots to the indicator
        typingIndicator.appendChild(dotsContainer);
        
        chatMessages.appendChild(typingIndicator);
        scrollToBottom();
    }

    function removeTypingIndicator() {
        const typingIndicator = document.getElementById('typingIndicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }

    function addBotWelcomeMessage(message) {
        const messageContainer = document.createElement('div');
        messageContainer.className = 'message-container';
        messageContainer.setAttribute('role', 'listitem');
        messageContainer.setAttribute('aria-label', 'Welcome message');

        const messageEl = document.createElement('div');
        messageEl.className = 'message bot-message';
        
        // Convert markdown-like syntax to HTML
        let formattedMessage = message
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') // Bold
            .replace(/\*(.*?)\*/g, '<em>$1</em>') // Italic
            .replace(/`(.*?)`/g, '<code>$1</code>') // Code
            .replace(/\n\n/g, '<br><br>') // Line breaks
            .replace(/\n/g, '<br>'); // Line breaks
        
        messageEl.innerHTML = formattedMessage;
        
        messageContainer.appendChild(messageEl);
        chatMessages.appendChild(messageContainer);
        
        // Add suggestion prompts
        addSuggestionPrompts();
        
        scrollToBottom();
    }

    function addSuggestionPrompts() {
        const suggestedPromptsContainer = document.createElement('div');
        suggestedPromptsContainer.className = 'suggested-prompts';
        suggestedPromptsContainer.id = 'suggestedPrompts';
        
        const grid = document.createElement('div');
        grid.className = 'prompts-grid';
        
        const suggestions = [
            { text: 'Can I get an abortion in my state?', emoji: '🗺️' },
            { text: 'What contraception methods are available?', emoji: '💊' },
            { text: 'How does pregnancy happen?', emoji: '🤰' },
            { text: 'What are some stress management tips?', emoji: '🧘' },
            { text: 'Explain STI prevention', emoji: '🛡️' },
            { text: 'What are the signs of pregnancy?', emoji: '🔍' }
        ];
        
        // Create rows with 2 buttons each
        for (let i = 0; i < suggestions.length; i += 2) {
            const row = document.createElement('div');
            row.className = 'prompt-row';
            
            // First button in row
            const btn1 = document.createElement('button');
            btn1.className = 'prompt-btn';
            btn1.setAttribute('data-prompt', suggestions[i].text);
            
            // Create text and emoji span for proper positioning
            const textSpan1 = document.createElement('span');
            textSpan1.className = 'prompt-text';
            textSpan1.textContent = suggestions[i].text;
            
            const emojiSpan1 = document.createElement('span');
            emojiSpan1.className = 'prompt-emoji';
            emojiSpan1.textContent = suggestions[i].emoji;
            
            btn1.appendChild(textSpan1);
            btn1.appendChild(emojiSpan1);
            row.appendChild(btn1);
            
            // Second button in row (if exists)
            if (i + 1 < suggestions.length) {
                const btn2 = document.createElement('button');
                btn2.className = 'prompt-btn';
                btn2.setAttribute('data-prompt', suggestions[i+1].text);
                
                const textSpan2 = document.createElement('span');
                textSpan2.className = 'prompt-text';
                textSpan2.textContent = suggestions[i+1].text;
                
                const emojiSpan2 = document.createElement('span');
                emojiSpan2.className = 'prompt-emoji';
                emojiSpan2.textContent = suggestions[i+1].emoji;
                
                btn2.appendChild(textSpan2);
                btn2.appendChild(emojiSpan2);
                row.appendChild(btn2);
            }
            
            grid.appendChild(row);
        }
        
        suggestedPromptsContainer.appendChild(grid);
        chatMessages.appendChild(suggestedPromptsContainer);
        
        // Add event listeners to the suggestion buttons
        const promptButtons = suggestedPromptsContainer.querySelectorAll('.prompt-btn');
        promptButtons.forEach(button => {
            button.addEventListener('click', function() {
                const promptText = this.getAttribute('data-prompt');
                sendMessage(promptText);
                
                // Hide suggestions after clicking
                suggestedPromptsContainer.style.display = 'none';
            });
        });
    }
    
    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function sendMessage(message) {
        addUserMessage(message);
        
        // Hide suggested prompts when user sends a message
        const suggestedPrompts = document.getElementById('suggestedPrompts');
        if (suggestedPrompts) {
            suggestedPrompts.style.display = 'none';
        }
        
        // Check for special command: "end"
        if (message.toLowerCase() === 'end') {
            // Show feedback dialog before ending session
            const feedbackDialog = document.createElement('div');
            feedbackDialog.className = 'feedback-dialog';
            feedbackDialog.id = 'end-session-feedback';
            
            const dialogContent = document.createElement('div');
            dialogContent.className = 'feedback-dialog-content';
            
            const title = document.createElement('h3');
            title.textContent = 'Before you go...';
            
            const text = document.createElement('p');
            text.textContent = 'Would you like to provide feedback on your experience?';
            
            const buttonsContainer = document.createElement('div');
            buttonsContainer.className = 'feedback-dialog-buttons';
            
            const provideFeedbackBtn = document.createElement('button');
            provideFeedbackBtn.className = 'provide-feedback-btn';
            provideFeedbackBtn.textContent = 'Provide Feedback';
            provideFeedbackBtn.addEventListener('click', function() {
                // Remove current dialog
                feedbackDialog.remove();
                
                // Show feedback form
                showFeedbackForm();
            });
            
            const skipFeedbackBtn = document.createElement('button');
            skipFeedbackBtn.className = 'skip-feedback-btn';
            skipFeedbackBtn.textContent = 'No Thanks';
            skipFeedbackBtn.addEventListener('click', function() {
                feedbackDialog.remove();
                clearSessionHistory();
            });
            
            buttonsContainer.appendChild(provideFeedbackBtn);
            buttonsContainer.appendChild(skipFeedbackBtn);
            
            dialogContent.appendChild(title);
            dialogContent.appendChild(text);
            dialogContent.appendChild(buttonsContainer);
            
            feedbackDialog.appendChild(dialogContent);
            document.body.appendChild(feedbackDialog);
            
            return;
        }
        
        // Normal message handling
        addTypingIndicator();
        
        fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                message: message,
                session_id: window.conversationId || null,
                user_location: window.userLocation || null
            })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            removeTypingIndicator();
            if (data.error) {
                addBotMessage("I'm sorry, but I encountered an error. Please try again in a moment.");
                console.error('API Error:', data.error);
            } else {
                // Store conversation ID if provided
                if (data.session_id) {
                    window.conversationId = data.session_id;
                }
                
                addBotMessage(
                    data.text, 
                    data.citations || [], 
                    data.citation_objects || null,
                    data.graphics || null
                );
            }
        })
        .catch(error => {
            removeTypingIndicator();
            console.error('Fetch Error:', error);
            addBotMessage("I'm sorry, but I couldn't connect to the server. Please check your internet connection and try again.");
        });
    }

    function submitFeedback(messageId, rating, comment = null) {
        fetch('/feedback', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message_id: messageId,
                rating: rating,
                comment: comment
            })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            console.log('Feedback submitted successfully:', data);
            // Hide the feedback form after submission
            const feedbackContainer = document.querySelector(`.feedback-container[data-message-id="${messageId}"]`);
            if (feedbackContainer) {
                feedbackContainer.innerHTML = '<div class="feedback-thanks">Thank you for your feedback!</div>';
            }
        })
        .catch(error => {
            console.error('Error submitting feedback:', error);
            alert('There was a problem submitting your feedback. Please try again.');
        });
    }

    // Focus input field on page load
    userInput.focus();

    // Clear session history
    function clearSessionHistory() {
        if (window.conversationId) {
            fetch(`/session`, {
                method: 'DELETE',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ 
                    session_id: window.conversationId 
                })
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to clear conversation history');
                }
                return response.json();
            })
            .then(data => {
                console.log('Session cleared successfully:', data);
                window.conversationId = null;
                
                // Reload the page to start fresh
                setTimeout(() => {
                    location.reload();
                }, 1000);
            })
            .catch(error => {
                console.error('Error clearing session:', error);
            });
        } else {
            // If no conversation ID, just reload
            location.reload();
        }
    }
});
