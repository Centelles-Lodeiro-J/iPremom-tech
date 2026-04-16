import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "iPremom Dashboard",
  description: "Optional clinician-facing dashboard for exported METABRIC pipeline summaries.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
