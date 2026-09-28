"use client";

import Link from "next/link";

export default function Error({ reset }: { error: Error & { digest?: string }; reset: () => void }) {
  return (
    <div className="min-h-dvh flex items-center justify-center bg-zinc-50 px-4">
      <div className="max-w-md text-center space-y-4">
        <h1 className="text-lg font-semibold text-zinc-900">Something went wrong</h1>
        <p className="text-sm text-zinc-600">
          This page hit an unexpected error. You can try again, or head back to the chat.
        </p>
        <div className="flex items-center justify-center gap-4">
          <button
            onClick={reset}
            className="rounded-full bg-blue-600 text-white px-5 py-2 text-sm font-medium focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2"
          >
            Try again
          </button>
          <Link href="/" className="text-sm text-blue-600 hover:underline">
            Back to chat
          </Link>
        </div>
      </div>
    </div>
  );
}
