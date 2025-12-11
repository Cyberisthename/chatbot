# JARVIS-2v Platform-Specific Deployment Guide

Step-by-step instructions for deploying JARVIS-2v to various cloud platforms.

---

## 📋 Table of Contents

1. [Vercel Deployment](#vercel-deployment)
2. [Netlify Deployment](#netlify-deployment)
3. [shiper.app Deployment](#shiperapp-deployment)
4. [Railway Deployment](#railway-deployment)
5. [Render Deployment](#render-deployment)
6. [DigitalOcean App Platform](#digitalocean-app-platform)
7. [AWS (Docker)](#aws-docker)

---

## Vercel Deployment

**Best for**: Frontend-only deployments (backend must be hosted separately)

### Prerequisites
- Vercel account (free tier available)
- Backend deployed elsewhere (Railway, Render, etc.)

### Steps

1. **Install Vercel CLI**
   ```bash
   npm install -g vercel
   ```

2. **Login to Vercel**
   ```bash
   vercel login
   ```

3. **Configure Environment**
   
   Edit `vercel.json` to set your backend URL:
   ```json
   {
     "framework": "nextjs",
     "buildCommand": "cd frontend && npm install && npm run build",
     "installCommand": "cd frontend && npm install",
     "outputDirectory": "frontend/.next"
   }
   ```

4. **Set Environment Variable**
   
   In Vercel dashboard or via CLI:
   ```bash
   vercel env add NEXT_PUBLIC_API_URL
   # Enter: https://your-backend-url.com
   ```

5. **Deploy**
   ```bash
   # Deploy to preview
   vercel
   
   # Deploy to production
   vercel --prod
   ```

6. **Verify**
   - Open the deployed URL
   - Check that Dashboard loads
   - Verify API connection in browser console

### Alternative: Via GitHub

1. Push code to GitHub
2. Import repository in Vercel dashboard
3. Set environment variables in Vercel project settings
4. Deploy automatically on push

---

## Netlify Deployment

**Best for**: Static frontend with backend proxy

### Prerequisites
- Netlify account (free tier available)
- Backend deployed elsewhere

### Steps

1. **Install Netlify CLI**
   ```bash
   npm install -g netlify-cli
   ```

2. **Login to Netlify**
   ```bash
   netlify login
   ```

3. **Configure netlify.toml**
   
   Already configured in the repo. Update backend URL:
   ```toml
   [build.environment]
     NEXT_PUBLIC_API_URL = "https://your-backend-url.com"
   ```

4. **Deploy**
   ```bash
   # Deploy to preview
   netlify deploy
   
   # Deploy to production
   netlify deploy --prod
   ```

5. **Set Environment Variables** (via dashboard)
   - Go to Site Settings → Environment Variables
   - Add `NEXT_PUBLIC_API_URL`: `https://your-backend-url.com`

6. **Verify**
   - Test the deployed URL
   - Check Network tab for API calls

### Alternative: Via GitHub

1. Push code to GitHub
2. Click "New site from Git" in Netlify
3. Select repository
4. Set build command: `cd frontend && npm run build`
5. Set publish directory: `frontend/.next`
6. Add environment variables
7. Deploy

---

## shiper.app Deployment

**Best for**: Full-stack Docker deployment

### Prerequisites
- shiper.app account
- Docker installed locally (for testing)

### Steps

1. **Test Docker Build Locally**
   ```bash
   docker-compose build
   docker-compose up
   # Verify at http://localhost:8000 and http://localhost:3000
   ```

2. **Create shiper.app Project**
   - Go to shiper.app dashboard
   - Create new project
   - Select "Docker Compose" deployment

3. **Configure Deployment**
   
   shiper.app will use the existing `docker-compose.yml`
   
   Ensure these environment variables are set:
   - `HOST=0.0.0.0`
   - `PORT=8000`
   - `NEXT_PUBLIC_API_URL=http://backend:8000`

4. **Push to Repository**
   ```bash
   git push origin main
   ```

5. **Deploy via shiper.app**
   - Connect GitHub repository
   - Select branch (main)
   - shiper.app will detect `docker-compose.yml`
   - Click "Deploy"

6. **Configure Networking**
   - Expose port 8000 (backend API)
   - Expose port 3000 (frontend)
   - Set up domain (optional)

7. **Verify**
   - Check backend: `https://your-app.shiper.app:8000/health`
   - Check frontend: `https://your-app.shiper.app:3000`

---

## Railway Deployment

**Best for**: Backend hosting with automatic deployments

### Deploy Backend to Railway

1. **Install Railway CLI**
   ```bash
   npm install -g @railway/cli
   ```

2. **Login**
   ```bash
   railway login
   ```

3. **Create New Project**
   ```bash
   railway init
   ```

4. **Deploy Backend**
   ```bash
   # From project root
   railway up
   ```

5. **Configure Service**
   - In Railway dashboard, go to your project
   - Add environment variables:
     - `HOST=0.0.0.0`
     - `PORT=${{PORT}}` (Railway provides this)
   - Set start command: `cd backend && python3 -m uvicorn main:app --host 0.0.0.0 --port $PORT`

6. **Get Backend URL**
   - Railway will provide a URL like `https://your-app.railway.app`
   - Copy this URL for frontend deployment

7. **Deploy Frontend to Vercel/Netlify**
   - Follow Vercel or Netlify steps above
   - Use Railway URL as `NEXT_PUBLIC_API_URL`

---

## Render Deployment

**Best for**: Full-stack deployment with separate services

### Deploy Backend

1. **Create New Web Service**
   - Go to Render dashboard
   - Click "New +" → "Web Service"
   - Connect GitHub repository

2. **Configure Backend Service**
   - Name: `jarvis-backend`
   - Root Directory: `backend`
   - Environment: `Python 3`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `cd .. && python3 -m uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
   - Plan: Free or Starter

3. **Add Environment Variables**
   - `HOST=0.0.0.0`
   - `PORT=${{PORT}}`
   - `PYTHONPATH=/opt/render/project/src`

4. **Deploy**
   - Render will auto-deploy on push
   - Get the URL (e.g., `https://jarvis-backend.onrender.com`)

### Deploy Frontend

1. **Create Static Site**
   - Click "New +" → "Static Site"
   - Connect same repository

2. **Configure Frontend Service**
   - Name: `jarvis-frontend`
   - Root Directory: `frontend`
   - Build Command: `npm install && npm run build`
   - Publish Directory: `frontend/.next`

3. **Add Environment Variable**
   - `NEXT_PUBLIC_API_URL=https://jarvis-backend.onrender.com`

4. **Deploy**
   - Frontend will build and deploy
   - Access at `https://jarvis-frontend.onrender.com`

---

## DigitalOcean App Platform

**Best for**: Managed container deployment

### Steps

1. **Create New App**
   - Go to DigitalOcean → Apps
   - Click "Create App"
   - Connect GitHub repository

2. **Configure Backend Component**
   - Detected as: Python
   - Build Command: `pip install -r backend/requirements.txt`
   - Run Command: `cd backend && python3 -m uvicorn main:app --host 0.0.0.0 --port 8080`
   - HTTP Port: 8080

3. **Configure Frontend Component**
   - Click "Add Component" → "Web Service"
   - Detected as: Node.js
   - Build Command: `cd frontend && npm install && npm run build`
   - Run Command: `cd frontend && npm start`
   - HTTP Port: 3000

4. **Environment Variables**
   
   Backend:
   - `HOST=0.0.0.0`
   - `PORT=8080`
   
   Frontend:
   - `NEXT_PUBLIC_API_URL=${backend.PUBLIC_URL}`

5. **Deploy**
   - Review and create
   - DigitalOcean will build and deploy both services

---

## AWS (Docker on ECS)

**Best for**: Enterprise deployments with AWS infrastructure

### Prerequisites
- AWS account
- AWS CLI installed and configured
- Docker installed

### Steps

1. **Create ECR Repository**
   ```bash
   aws ecr create-repository --repository-name jarvis-2v
   ```

2. **Build and Push Docker Image**
   ```bash
   # Login to ECR
   aws ecr get-login-password --region us-east-1 | \
     docker login --username AWS --password-stdin YOUR_ECR_URL
   
   # Build image
   docker build -t jarvis-2v:latest .
   
   # Tag image
   docker tag jarvis-2v:latest YOUR_ECR_URL/jarvis-2v:latest
   
   # Push image
   docker push YOUR_ECR_URL/jarvis-2v:latest
   ```

3. **Create ECS Cluster**
   ```bash
   aws ecs create-cluster --cluster-name jarvis-cluster
   ```

4. **Create Task Definition**
   
   Create `task-definition.json`:
   ```json
   {
     "family": "jarvis-2v",
     "containerDefinitions": [
       {
         "name": "jarvis-backend",
         "image": "YOUR_ECR_URL/jarvis-2v:latest",
         "portMappings": [
           {"containerPort": 8000, "hostPort": 8000}
         ],
         "environment": [
           {"name": "HOST", "value": "0.0.0.0"},
           {"name": "PORT", "value": "8000"}
         ]
       }
     ]
   }
   ```
   
   Register task:
   ```bash
   aws ecs register-task-definition --cli-input-json file://task-definition.json
   ```

5. **Create ECS Service**
   ```bash
   aws ecs create-service \
     --cluster jarvis-cluster \
     --service-name jarvis-service \
     --task-definition jarvis-2v \
     --desired-count 1 \
     --launch-type FARGATE
   ```

6. **Set Up Load Balancer** (optional)
   - Create Application Load Balancer in AWS Console
   - Configure target groups for ports 8000 and 3000
   - Update ECS service to use load balancer

---

## Comparison Matrix

| Platform | Frontend | Backend | Docker | Free Tier | Best For |
|----------|----------|---------|--------|-----------|----------|
| Vercel | ✅ | ❌ | ❌ | Yes | Next.js apps |
| Netlify | ✅ | ⚠️ (Functions) | ❌ | Yes | JAMstack |
| shiper.app | ✅ | ✅ | ✅ | Varies | Full Docker |
| Railway | ✅ | ✅ | ✅ | Limited | Quick deploy |
| Render | ✅ | ✅ | ✅ | Yes | Full-stack |
| DigitalOcean | ✅ | ✅ | ✅ | $5/mo | Managed apps |
| AWS ECS | ✅ | ✅ | ✅ | Limited | Enterprise |

---

## Cost Estimates (as of 2024)

| Platform | Free Tier | Paid Plan |
|----------|-----------|-----------|
| Vercel | Yes (Hobby) | $20/mo (Pro) |
| Netlify | Yes (Starter) | $19/mo (Pro) |
| shiper.app | Varies | Contact |
| Railway | $5 credit/mo | Pay as you go |
| Render | Yes (Limited) | $7/mo (Starter) |
| DigitalOcean | None | $5/mo (Basic) |
| AWS | 12 months | Variable |

---

## Recommended Stack

### For Development
- Local: `./scripts/start_all_local.sh`
- Docker: `docker-compose up`

### For Production

**Option 1: Separate Services (Recommended)**
- Frontend: Vercel (free, fast, auto-deploys)
- Backend: Railway or Render (managed, auto-scaling)

**Option 2: All-in-One**
- shiper.app or DigitalOcean (Docker Compose, easy setup)

**Option 3: Enterprise**
- AWS ECS + CloudFront (scalable, reliable)

---

## Post-Deployment Checklist

After deploying to any platform:

- [ ] Test health endpoint: `curl https://your-backend/health`
- [ ] Open frontend URL in browser
- [ ] Check Dashboard loads
- [ ] Run an inference test
- [ ] Create an adapter
- [ ] Run a quantum experiment
- [ ] Check logs for errors
- [ ] Set up monitoring (optional)
- [ ] Configure custom domain (optional)
- [ ] Enable HTTPS (should be automatic on most platforms)

---

## Troubleshooting

### Frontend can't connect to backend
- Check `NEXT_PUBLIC_API_URL` is set correctly
- Verify CORS is enabled on backend (it is by default)
- Check backend is accessible: `curl https://your-backend/health`

### Backend crashes on startup
- Check logs for Python errors
- Verify all dependencies installed
- Check environment variables are set
- Ensure sufficient memory (512MB minimum)

### Build fails
- Check Node.js version is 18+
- Check Python version is 3.10+
- Clear build cache and retry
- Check disk space is sufficient

---

## Support

For deployment issues:
1. Check the logs on your platform
2. Refer to `TESTING_GUIDE.md` for troubleshooting
3. Review platform-specific documentation
4. Open an issue on GitHub with logs

Happy deploying! 🚀
