import express from 'express';
import cors from 'cors';
import axios from 'axios';
import ytdl from 'ytdl-core';
import multer from 'multer';
import path from 'path';
import fs from 'fs';

const app = express();
const port = 3001;

// Configure multer for handling file uploads
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, 'uploads/');
  },
  filename: function (req, file, cb) {
    cb(null, Date.now() + path.extname(file.originalname));
  }
});

const upload = multer({ 
  storage: storage,
  limits: {
    fileSize: 100 * 1024 * 1024, // 100MB limit
  }
});

// Create uploads directory if it doesn't exist
if (!fs.existsSync('uploads')) {
  fs.mkdirSync('uploads');
}

app.use(cors());
app.use(express.json({ limit: '100mb' }));
app.use(express.urlencoded({ extended: true, limit: '100mb' }));
app.use('/uploads', express.static('uploads'));

// Helper function to validate YouTube URL
const isValidYouTubeUrl = (url) => {
  try {
    const parsed = new URL(url);
    return (
      parsed.hostname === 'www.youtube.com' ||
      parsed.hostname === 'youtube.com' ||
      parsed.hostname === 'youtu.be' ||
      parsed.hostname === 'm.youtube.com'
    );
  } catch (error) {
    return false;
  }
};

app.get('/api/proxy-image', async (req, res) => {
  try {
    const { url } = req.query;
    if (!url) {
      return res.status(400).send('URL parameter is required');
    }

    const response = await axios({
      url,
      responseType: 'arraybuffer',
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
      }
    });

    const contentType = response.headers['content-type'];
    res.setHeader('Content-Type', contentType);
    res.send(response.data);
  } catch (error) {
    console.error('Error proxying image:', error);
    res.status(500).send('Failed to fetch image');
  }
});

app.get('/api/youtube', async (req, res) => {
  try {
    console.log('Received YouTube request:', req.query);
    const { url } = req.query;
    
    if (!url) {
      return res.status(400).send('URL parameter is required');
    }

    if (!isValidYouTubeUrl(url)) {
      return res.status(400).send('Invalid YouTube URL');
    }

    // Convert shorts URL to regular URL if needed
    const videoId = url.includes('/shorts/') 
      ? url.split('/shorts/')[1].split('?')[0]
      : url.includes('v=') 
        ? url.split('v=')[1].split('&')[0]
        : null;

    if (!videoId) {
      return res.status(400).send('Could not extract video ID');
    }

    console.log('Processing video ID:', videoId);

    try {
      // Get video info
      console.log('Fetching video info...');
      const info = await ytdl.getInfo(videoId);
      
      // Extract relevant information
      const videoInfo = {
        title: info.videoDetails.title,
        lengthSeconds: info.videoDetails.lengthSeconds,
        thumbnail: info.videoDetails.thumbnails[info.videoDetails.thumbnails.length - 1].url,
        author: info.videoDetails.author.name,
        viewCount: info.videoDetails.viewCount,
        videoId: videoId
      };

      console.log('Video info retrieved:', videoInfo);

      // Return video information
      res.json(videoInfo);

    } catch (error) {
      console.error('Error in video processing:', error);
      if (!res.headersSent) {
        res.status(500).send('Error processing video: ' + error.message);
      }
    }
  } catch (error) {
    console.error('Error in YouTube endpoint:', error);
    if (!res.headersSent) {
      res.status(500).send('Error: ' + error.message);
    }
  }
});

// File upload endpoint
app.post('/api/upload', upload.single('file'), (req, res) => {
  console.log('Received upload request');
  try {
    if (!req.file) {
      console.error('No file in request');
      return res.status(400).json({ error: 'No file uploaded' });
    }
    console.log('File uploaded successfully:', {
      filename: req.file.filename,
      mimetype: req.file.mimetype,
      size: req.file.size
    });
    res.json({ 
      success: true,
      file: {
        path: `/uploads/${req.file.filename}`,
        mimetype: req.file.mimetype,
        size: req.file.size
      }
    });
  } catch (error) {
    console.error('Upload error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error('Global error handler:', err);
  if (err instanceof multer.MulterError) {
    if (err.code === 'LIMIT_FILE_SIZE') {
      return res.status(400).json({ error: 'File is too large' });
    }
    return res.status(400).json({ error: err.message });
  }
  res.status(500).json({ error: err.message });
});

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});
