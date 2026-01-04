"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Home, Cpu, Atom, Terminal, Settings } from "lucide-react";

const navItems = [
  { href: "/", label: "Dashboard", icon: Home },
  { href: "/adapters", label: "Adapters", icon: Cpu },
  { href: "/quantum", label: "Quantum Lab", icon: Atom },
  { href: "/console", label: "Console", icon: Terminal },
  { href: "/settings", label: "Settings", icon: Settings },
];

export default function Navigation() {
  const pathname = usePathname();

  return (
    <nav className="border-b border-jarvis bg-jarvis-dark/50 backdrop-blur-md sticky top-0 z-50">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <Link href="/" className="flex items-center space-x-2">
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-jarvis-primary to-jarvis-accent flex items-center justify-center text-black font-bold text-xl glow">
              J
            </div>
            <span className="text-xl font-bold text-jarvis">JARVIS-2v</span>
          </Link>

          <div className="flex space-x-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = pathname === item.href;
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all ${
                    isActive
                      ? "bg-jarvis-primary/20 text-jarvis-primary border border-jarvis-primary/50"
                      : "text-gray-400 hover:text-jarvis-primary hover:bg-jarvis-primary/10"
                  }`}
                >
                  <Icon size={18} />
                  <span className="hidden md:inline">{item.label}</span>
                </Link>
              );
            })}
          </div>
        </div>
      </div>
    </nav>
  );
}
