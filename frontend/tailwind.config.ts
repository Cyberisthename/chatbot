import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        jarvis: {
          primary: "#00d4ff",
          secondary: "#0088cc",
          dark: "#0a0f1c",
          darker: "#060911",
          accent: "#00ffaa",
        },
      },
    },
  },
  plugins: [],
};

export default config;
