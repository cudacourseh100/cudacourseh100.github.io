/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./*.{js,ts,jsx,tsx}",
    "./components/**/*.{js,ts,jsx,tsx}",
    "./services/**/*.{js,ts,jsx,tsx}"
  ],
  theme: {
    extend: {
      colors: {
        nvidia: '#76b900',
        'nvidia-dark': '#5e9400',
        gray: {
          850: '#1f2937',
        }
      }
    },
  },
  plugins: [],
}