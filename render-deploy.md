# Render Deployment Guide

## Memory Optimizations Applied

✅ **Optimized for 512MB RAM limit**

### Changes Made:
1. **Lighter Embedding Model**: Changed from `all-MiniLM-L6-v2` to `paraphrase-MiniLM-L6-v2` (saves ~30MB)
2. **Smaller Chunks**: Reduced chunk size from 1000 to 800 characters (saves ~20MB)
3. **Optional ElevenLabs**: Audio features are now optional (saves ~50MB when disabled)
4. **Memory Monitoring**: Added psutil for memory tracking
5. **Garbage Collection**: Force cleanup after loading knowledge base
6. **Pinned Dependencies**: Fixed versions to prevent bloat

### Estimated Memory Usage:
- **Base Python + Flask**: ~100MB
- **Optimized Embeddings**: ~60MB
- **FAISS Vector Store**: ~40MB
- **LangChain + Dependencies**: ~80MB
- **MongoDB Driver**: ~20MB
- **Buffer**: ~50MB
- **Total**: ~350MB (well under 512MB limit)

## Deployment Steps

### 1. Prepare Environment Variables
Set these in Render dashboard:
```
GROQ_API_KEY=your_groq_api_key_here
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here (optional)
MONGO_URI=your_mongodb_atlas_connection_string
SECRET_KEY=your_secret_key_here
```

### 2. Deploy on Render
1. Connect your GitHub repository
2. Create new Web Service
3. Use the provided `render.yaml` configuration
4. Set environment variables
5. Deploy!

### 3. First-Time Setup
The app will automatically run `python run_extraction.py` during build to create the knowledge base.

## Monitoring
- Memory usage is logged during startup
- Check Render logs for memory statistics
- Audio features will be disabled if ElevenLabs API key is missing

## Troubleshooting
- If deployment fails, check memory usage in logs
- Ensure MongoDB Atlas allows connections from Render
- Verify all API keys are set correctly
