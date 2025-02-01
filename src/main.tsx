/**
 * Main Entry Point - AIorNOT Application
 * 
 * @author T. Landon Love <12stonedesigns@gmail.com>
 * @copyright Copyright (c) 2024 T. Landon Love
 * @license MIT
 */

import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App.tsx';
import './index.css';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>
);
