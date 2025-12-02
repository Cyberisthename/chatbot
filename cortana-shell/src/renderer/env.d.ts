import type { CortanaAPI } from '../shared/types/cortana';

declare global {
  interface Window {
    cortana: CortanaAPI;
  }
}

export {};
