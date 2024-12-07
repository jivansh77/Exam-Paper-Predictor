/** @type {import('tailwindcss').Config} */
import '@tailwindcss/forms'
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [
    '@tailwindcss/forms',
  ],
}