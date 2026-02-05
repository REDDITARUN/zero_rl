import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./hooks/**/*.{js,ts,jsx,tsx,mdx}",
    "./lib/**/*.{js,ts,jsx,tsx,mdx}"
  ],
  theme: {
    extend: {
      colors: {
        ink: "#2f2a23",
        sand: "#f2ecdf",
        clay: "#d8c7ab",
        forest: "#3f5a4b",
        moss: "#7a8e63"
      },
      boxShadow: {
        soft: "0 10px 30px rgba(47, 42, 35, 0.08)"
      }
    }
  },
  plugins: []
};

export default config;
