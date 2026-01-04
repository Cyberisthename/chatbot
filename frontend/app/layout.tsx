import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Navigation from "@/components/Navigation";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "JARVIS-2v | Modular Edge AI",
  description: "Modular AI System with Adapter Engine & Synthetic Quantum Lab",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <div className="min-h-screen flex flex-col">
          <Navigation />
          <main className="flex-1 container mx-auto px-4 py-8">
            {children}
          </main>
          <footer className="border-t border-jarvis py-4 text-center text-sm text-gray-500">
            <p>JARVIS-2v © 2024 | Modular Edge AI System</p>
          </footer>
        </div>
      </body>
    </html>
  );
}
