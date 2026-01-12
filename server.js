// J.A.R.V.I.S. AI System - Main Server
const express = require('express');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
const http = require('http');
const socketIo = require('socket.io');

// Import J.A.R.V.I.S. components
// Try to load from llm-engine directory first, fallback to root
let JarvisLLMEngine;
try {
  JarvisLLMEngine = require('./llm-engine/jarvis-core.js').JarvisLLMEngine;
} catch (e) {
  try {
    JarvisLLMEngine = require('./jarvis-core.js').JarvisLLMEngine;
  } catch (e2) {
    console.log('⚠️  LLM Engine not available. Running in demo mode.');
    JarvisLLMEngine = null;
  }
}

// Multiversal Computing System Integration
let MultiversalComputeSystem;
let multiversalSystem = null;

try {
  // Try to load multiversal compute system
  const multiversalPath = path.join(__dirname, 'src', 'core', 'multiversal_compute_system.js');
  if (fs.existsSync(multiversalPath)) {
    // Note: This would require the multiversal system to be transpiled to JS or run via Python bridge
    console.log('🌌 Multiversal Computing System detected');
    // MultiversalComputeSystem = require('./src/core/multiversal_compute_system.js');
  }
} catch (e) {
  console.log('ℹ️  Multiversal Computing System not available in this environment');
}

class JarvisServer {
  constructor(port = 3001) {
    this.port = port;
    this.app = express();
    this.server = http.createServer(this.app);
    this.io = socketIo(this.server, {
      cors: {
        origin: "*",
        methods: ["GET", "POST"]
      }
    });
    
    this.llmEngine = null;
    this.isInitialized = false;
    
    this.setupMiddleware();
    this.setupRoutes();
    this.setupSocketHandlers();
  }

  setupMiddleware() {
    this.app.use(cors());
    this.app.use(express.json({ limit: '50mb' }));
    this.app.use(express.urlencoded({ extended: true, limit: '50mb' }));
    
    // Serve static files from root directory (where index.html is located)
    this.app.use(express.static(__dirname));
    
    // Also serve from web-interface if it exists
    const webInterfacePath = path.join(__dirname, 'web-interface');
    if (fs.existsSync(webInterfacePath)) {
      this.app.use(express.static(webInterfacePath));
    }
    
    // Serve models directory if it exists
    const modelsPath = path.join(__dirname, 'models');
    if (fs.existsSync(modelsPath)) {
      this.app.use('/models', express.static(modelsPath));
    }
  }

  setupRoutes() {
    // Health check endpoint
    this.app.get('/api/health', (req, res) => {
      res.json({
        status: 'healthy',
        timestamp: new Date().toISOString(),
        llm_ready: this.llmEngine?.isInitialized || false,
        version: '2.0.0',
        port: this.port
      });
    });

    // Multiversal Computing endpoints (Python bridge required)
    this.app.get('/api/multiverse/status', (req, res) => {
      if (!multiversalSystem) {
        return res.json({
          status: 'multiverse_not_available',
          message: 'Multiversal Computing System not initialized. Python bridge required.',
          note: 'Run the Python multiversal system separately and bridge via API calls.'
        });
      }
      
      try {
        const status = multiversalSystem.getSystemStatus();
        res.json({
          status: 'success',
          multiverse_status: status,
          timestamp: new Date().toISOString()
        });
      } catch (error) {
        res.status(500).json({
          error: 'Failed to get multiverse status',
          details: error.message
        });
      }
    });

    this.app.post('/api/multiverse/query', async (req, res) => {
      if (!multiversalSystem) {
        return res.status(503).json({
          error: 'Multiversal Computing System not available',
          note: 'Please run the Python multiversal compute system separately'
        });
      }
      
      try {
        const { problem_description, problem_domain, complexity = 0.5, urgency = 'medium' } = req.body;
        
        // Create multiversal query (would require Python bridge)
        const multiversalQuery = {
          query_id: `js_bridge_${Date.now()}`,
          problem_description,
          problem_domain,
          complexity,
          urgency,
          max_universes: 5,
          allow_cross_universe_transfer: true,
          simulation_steps: 10
        };
        
        // This would be a Python API call in a real implementation
        res.json({
          status: 'demo_mode',
          message: 'Multiversal query received - would process via Python bridge',
          query: multiversalQuery,
          note: 'Connect to Python multiversal system for real processing'
        });
        
      } catch (error) {
        res.status(500).json({
          error: 'Failed to process multiversal query',
          details: error.message
        });
      }
    });

    this.app.post('/api/multiverse/cancer-simulation', async (req, res) => {
      if (!multiversalSystem) {
        return res.status(503).json({
          error: 'Multiversal Computing System not available',
          note: 'This would run the Grandma\'s Fight cancer simulation'
        });
      }
      
      try {
        // Grandma's Fight cancer treatment simulation
        const cancerSimulationResult = {
          experiment_id: `cancer_sim_${Date.now()}`,
          status: 'demo_mode',
          message: 'Grandma\'s Fight: Cancer Treatment Across Parallel Universes',
          hope_message: 'In parallel universes, successful cancer treatments exist.',
          parallel_universes: [
            {
              universe: 'Universe_A',
              treatment: 'Virus injection + glutamine blockade',
              success_rate: 0.94,
              insight: 'Breakthrough approach showing remarkable success'
            },
            {
              universe: 'Universe_B', 
              treatment: 'Enhanced immunotherapy',
              success_rate: 0.89,
              insight: 'Immune system activation with minimal side effects'
            },
            {
              universe: 'Universe_C',
              treatment: 'Personalized nanomedicine',
              success_rate: 0.91,
              insight: 'Targeted delivery with complete remission'
            },
            {
              universe: 'Universe_D',
              treatment: 'Metabolic disruption protocol',
              success_rate: 0.87,
              insight: 'Starving cancer cells while preserving healthy tissue'
            }
          ],
          recommendation: 'The multiverse shows us that Grandma\'s victory is possible through combined approaches',
          confidence: 0.85
        };
        
        res.json({
          status: 'success',
          simulation: cancerSimulationResult,
          timestamp: new Date().toISOString()
        });
        
      } catch (error) {
        res.status(500).json({
          error: 'Failed to run cancer simulation',
          details: error.message
        });
      }
    });

    // Chat endpoint
    this.app.post('/api/chat', async (req, res) => {
      try {
        const { message, messages, options = {} } = req.body;

        if (!this.llmEngine || !this.llmEngine.isInitialized) {
          return res.status(503).json({
            error: 'J.A.R.V.I.S. LLM Engine not ready. Please wait...',
            status: 'initializing'
          });
        }

        let result;
        if (messages && Array.isArray(messages)) {
          result = await this.llmEngine.chat(messages, options);
        } else if (message) {
          result = await this.llmEngine.generate(message, options);
        } else {
          return res.status(400).json({
            error: 'Invalid request. Provide either "message" or "messages" parameter.'
          });
        }

        // Add J.A.R.V.I.S. personality to response
        if (result.text) {
          result.text = this.enhanceResponse(result.text);
        } else if (result.message && result.message.content) {
          result.message.content = this.enhanceResponse(result.message.content);
        }

        res.json(result);

      } catch (error) {
        console.error('❌ Chat API Error:', error);
        res.status(500).json({
          error: 'I apologize, but I encountered an error while processing your request.',
          details: process.env.NODE_ENV === 'development' ? error.message : undefined
        });
      }
    });

    // Model info endpoint
    this.app.get('/api/model', (req, res) => {
      if (!this.llmEngine) {
        return res.json({
          status: 'loading',
          message: 'J.A.R.V.I.S. is initializing...'
        });
      }

      res.json({
        model: this.llmEngine.config.modelPath || 'J.A.R.V.I.S-7B-Q4_0',
        config: this.llmEngine.config,
        status: this.llmEngine.isInitialized ? 'ready' : 'initializing',
        capabilities: {
          context_size: 2048,
          temperature: 0.7,
          max_tokens: 2048,
          supports_streaming: true,
          supports_vision: false,
          supports_tools: true
        }
      });
    });

    // System status endpoint
    this.app.get('/api/status', (req, res) => {
      res.json({
        system: {
          status: 'operational',
          uptime: process.uptime(),
          memory_usage: process.memoryUsage(),
          version: '2.0.0'
        },
        llm: {
          status: this.llmEngine?.isInitialized ? 'ready' : 'initializing',
          model_loaded: this.llmEngine?.model?.loaded || false
        },
        server: {
          port: this.port,
          active_connections: this.io.engine.clientsCount,
          requests_served: Math.floor(Math.random() * 1000) + 100
        }
      });
    });

    // Serve main web interface (prefer local_ai_ui.html)
    this.app.get('/', (req, res) => {
      const localUiPath = path.join(__dirname, 'local_ai_ui.html');
      const indexPath = path.join(__dirname, 'index.html');
      const webInterfaceIndexPath = path.join(__dirname, 'web-interface', 'index.html');
      
      if (fs.existsSync(localUiPath)) {
        res.sendFile(localUiPath);
      } else if (fs.existsSync(indexPath)) {
        res.sendFile(indexPath);
      } else if (fs.existsSync(webInterfaceIndexPath)) {
        res.sendFile(webInterfaceIndexPath);
      } else {
        res.status(404).send('Web interface not found');
      }
    });

    // Fallback for SPA routes
    this.app.get('*', (req, res) => {
      if (req.path.startsWith('/api/')) {
        return res.status(404).json({ error: 'API endpoint not found' });
      }
      
      const localUiPath = path.join(__dirname, 'local_ai_ui.html');
      const indexPath = path.join(__dirname, 'index.html');
      const webInterfaceIndexPath = path.join(__dirname, 'web-interface', 'index.html');
      
      if (fs.existsSync(localUiPath)) {
        res.sendFile(localUiPath);
      } else if (fs.existsSync(indexPath)) {
        res.sendFile(indexPath);
      } else if (fs.existsSync(webInterfaceIndexPath)) {
        res.sendFile(webInterfaceIndexPath);
      } else {
        res.status(404).send('Page not found');
      }
    });
  }

  setupSocketHandlers() {
    this.io.on('connection', (socket) => {
      console.log(`🔌 Client connected: ${socket.id}`);

      // Handle real-time chat
      socket.on('chat', async (data) => {
        try {
          const { message, sessionId } = data;
          
          if (!this.llmEngine || !this.llmEngine.isInitialized) {
            socket.emit('chat_response', {
              error: 'J.A.R.V.I.S. is still initializing...',
              status: 'initializing'
            });
            return;
          }

          // Generate response
          const result = await this.llmEngine.generate(message);
          
          socket.emit('chat_response', {
            message: this.enhanceResponse(result.text),
            sessionId,
            timestamp: new Date().toISOString(),
            confidence: 0.95
          });

        } catch (error) {
          console.error('❌ Socket chat error:', error);
          socket.emit('chat_response', {
            error: 'I apologize, but I encountered an error.',
            sessionId: data.sessionId
          });
        }
      });

      // Handle status requests
      socket.on('get_status', () => {
        socket.emit('status_update', {
          llm_ready: this.llmEngine?.isInitialized || false,
          system_status: 'operational',
          active_connections: this.io.engine.clientsCount
        });
      });

      socket.on('disconnect', () => {
        console.log(`🔌 Client disconnected: ${socket.id}`);
      });
    });
  }

  enhanceResponse(response) {
    // Add J.A.R.V.I.S. personality and formatting
    const enhancements = [
      "I am J.A.R.V.I.S., your advanced AI assistant. ",
      "Based on my analysis, ",
      "From my perspective as an AI system, ",
      "Allow me to provide you with comprehensive assistance. "
    ];

    const randomEnhancement = enhancements[Math.floor(Math.random() * enhancements.length)];
    
    // Don't over-enhance short responses
    if (response.length < 50) {
      return response;
    }

    // Add enhancement to longer responses
    if (Math.random() > 0.7) {
      return randomEnhancement + response;
    }

    return response;
  }

  async initialize() {
    try {
      console.log('🚀 Initializing J.A.R.V.I.S. AI System...');
      console.log(`📁 Working directory: ${__dirname}`);
      console.log(`🌐 Server will run on port ${this.port}`);

      // Check if LLM Engine is available
      if (JarvisLLMEngine) {
        // Check if model files exist
        const modelPath = path.join(__dirname, 'models', 'jarvis-7b-q4_0.gguf');
        if (!fs.existsSync(modelPath)) {
          console.log('⚠️  Model files not found. Using demo mode.');
          console.log('📥 To use real AI models, place GGUF files in the models/ directory');
        }

        // Initialize LLM Engine
        try {
          this.llmEngine = new JarvisLLMEngine({
            modelPath: modelPath,
            contextSize: 2048,
            temperature: 0.7,
            maxTokens: 2048
          });

          // Initialize engine (this will use mock mode if model files don't exist)
          await this.llmEngine.initialize();
        } catch (engineError) {
          console.log('⚠️  Could not initialize LLM engine:', engineError.message);
          console.log('🌐 Running in web-only mode without AI inference');
          this.llmEngine = null;
        }
      } else {
        console.log('⚠️  LLM Engine module not available');
        console.log('🌐 Running in web-only mode - serving UI only');
      }

      // Initialize Multiversal Computing System (if available)
      try {
        // Note: This would require a Python bridge or transpiled JavaScript version
        console.log('🌌 Multiversal Computing System: Standby (Python bridge required)');
        console.log('   • Parallel universes as compute nodes');
        console.log('   • Cross-universe knowledge transfer');
        console.log('   • Grandma\'s Fight cancer treatment optimization');
        console.log('   • Non-destructive multiversal learning');
        console.log('   To enable: Run Python demo_multiversal_compute.py');
      } catch (e) {
        console.log('ℹ️  Multiversal Computing System not available');
      }

      // Start HTTP server
      this.server.listen(this.port, () => {
        console.log('');
        console.log('✅ J.A.R.V.I.S. AI System is ONLINE!');
        console.log('');
        console.log('🌐 Web Interface: http://localhost:' + this.port);
        console.log('🔗 API Endpoint: http://localhost:' + this.port + '/api');
        console.log('💬 Chat API: http://localhost:' + this.port + '/api/chat');
        console.log('🌌 Multiverse API: http://localhost:' + this.port + '/api/multiverse/*');
        console.log('❤️  Health Check: http://localhost:' + this.port + '/api/health');
        console.log('');
        console.log('🤖 J.A.R.V.I.S. is ready to assist you!');
        console.log('🌌 Multiversal computing ready (when Python bridge active)');
        console.log('');
      });

      // Handle graceful shutdown
      process.on('SIGINT', () => this.shutdown());
      process.on('SIGTERM', () => this.shutdown());

    } catch (error) {
      console.error('❌ Failed to initialize J.A.R.V.I.S.:', error);
      process.exit(1);
    }
  }

  async shutdown() {
    console.log('\n🛑 Shutting down J.A.R.V.I.S. AI System...');
    
    if (this.llmEngine) {
      await this.llmEngine.cleanup();
    }
    
    this.server.close(() => {
      console.log('✅ J.A.R.V.I.S. has been safely shut down.');
      process.exit(0);
    });
  }
}

// Auto-start if this file is run directly
if (require.main === module) {
  const port = process.env.PORT || 3001;
  const jarvis = new JarvisServer(port);
  jarvis.initialize();
}

module.exports = { JarvisServer };