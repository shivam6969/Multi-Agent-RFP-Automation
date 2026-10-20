/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["ui-monospace", "SFMono-Regular", "Consolas", "monospace"],
      },
      colors: {
        // Violet brand colour from the designs.
        brand: {
          50: "#f4f2ff",
          100: "#ebe7ff",
          200: "#d9d2fe",
          300: "#bdb1fc",
          400: "#9d89f8",
          500: "#7c66f1",
          600: "#6a4fe4",
          700: "#5a3fcb",
          800: "#4b35a6",
          900: "#3f2f85",
        },
        // Dark "grand total" card.
        cocoa: { 800: "#4a3c3d", 900: "#3b3031" },
      },
      boxShadow: { card: "0 1px 2px rgba(16,24,40,.04), 0 1px 3px rgba(16,24,40,.06)" },
    },
  },
  plugins: [],
};
