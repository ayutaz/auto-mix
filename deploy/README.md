# AutoMix Cloud Deployment Guide

This guide explains how to deploy AutoMix to various cloud platforms.

## Prerequisites

- Docker installed locally for testing
- Account on your chosen cloud platform
- Git repository with your code

## Local Testing with Docker

1. Build and run locally:
```bash
docker-compose up --build
```

2. Access the application at http://localhost:5000

## Deployment Options

### Option 1: Railway (Recommended for simplicity)

Railway provides the easiest deployment experience with automatic builds and SSL.

1. Install Railway CLI:
```bash
npm install -g @railway/cli
```

2. Login and initialize:
```bash
railway login
railway init
```

3. Deploy:
```bash
railway up
```

4. Set environment variables in Railway dashboard:
   - `FLASK_ENV=production`

### Option 2: Render

Render offers good performance with automatic scaling and zero-downtime deploys.

1. Connect your GitHub repository to Render
2. Create a new Web Service
3. Select "Docker" as the environment
4. Use the provided `render.yaml` configuration
5. Deploy

### Option 3: Fly.io

Fly.io provides global edge deployment with excellent performance.

1. Install Fly CLI:
```bash
curl -L https://fly.io/install.sh | sh
```

2. Login and create app:
```bash
fly auth login
fly launch
```

3. Deploy:
```bash
fly deploy
```

### Option 4: DigitalOcean App Platform

1. Create a new App in DigitalOcean
2. Connect your GitHub repository
3. Choose "Dockerfile" as the build type
4. Configure environment variables:
   - `FLASK_ENV=production`
   - `PORT=8080` (DigitalOcean uses 8080)
5. Deploy

### Option 5: Google Cloud Run

1. Install gcloud CLI and authenticate
2. Build and push image:
```bash
gcloud builds submit --tag gcr.io/PROJECT-ID/automix
```

3. Deploy:
```bash
gcloud run deploy automix \
  --image gcr.io/PROJECT-ID/automix \
  --platform managed \
  --allow-unauthenticated \
  --port 5000
```

### Option 6: AWS App Runner

1. Build and push to ECR:
```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin [your-ecr-uri]
docker build -t automix .
docker tag automix:latest [your-ecr-uri]/automix:latest
docker push [your-ecr-uri]/automix:latest
```

2. Create App Runner service in AWS Console
3. Configure to use your ECR image

## Storage Considerations

For production use, you'll need to configure cloud storage for uploaded files:

### AWS S3 / Cloudflare R2
1. Create a bucket
2. Set environment variables:
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`
   - `S3_BUCKET_NAME`
   - For R2: `S3_ENDPOINT_URL`

### Google Cloud Storage
1. Create a bucket
2. Set up service account
3. Set `GOOGLE_APPLICATION_CREDENTIALS`

## Performance Optimization

1. **Use a CDN**: CloudFlare or AWS CloudFront for static assets
2. **Add Redis**: For caching and job queues
3. **Use a job queue**: Celery or RQ for background processing
4. **Scale horizontally**: Most platforms support auto-scaling

## Security Considerations

1. Always use HTTPS (most platforms provide this automatically)
2. Set secure headers
3. Implement rate limiting
4. Use environment variables for secrets
5. Regular security updates

## Monitoring

1. Set up error tracking (Sentry)
2. Configure logging (Papertrail, Loggly)
3. Add performance monitoring (New Relic, DataDog)
4. Set up uptime monitoring (UptimeRobot, Pingdom)

## Cost Estimates

- **Railway**: ~$5-20/month for starter
- **Render**: Free tier available, ~$7/month for standard
- **Fly.io**: ~$5-15/month for small instances
- **DigitalOcean**: ~$12/month for basic droplet
- **Google Cloud Run**: Pay per request, ~$10-50/month
- **AWS App Runner**: ~$10-30/month

## Troubleshooting

### Memory Issues
- Enable chunk processing mode
- Use streaming mode for large files
- Increase instance memory

### Slow Processing
- Use GPU instances if available
- Enable preview mode for testing
- Implement job queues

### File Upload Issues
- Check file size limits
- Verify storage permissions
- Monitor disk space

### Connection Timeouts
- Increase timeout settings
- Use background jobs
- Implement progress updates