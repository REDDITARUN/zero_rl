import { ReactNode } from "react";

interface MessageListProps {
  children: ReactNode;
}

export function MessageList({ children }: MessageListProps) {
  return <div className="scrollbar-thin h-full overflow-y-auto px-3 pb-3">{children}</div>;
}
