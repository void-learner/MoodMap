/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx,ts,tsx}",  // Scans all JS/TS files in src/ for Tailwind classes
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}

