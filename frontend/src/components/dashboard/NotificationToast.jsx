import { X } from "lucide-react";

export default function NotificationToast({ notifications, onDismiss }) {
  if (!notifications.length) return null;

  return (
    <div className="toast-stack" role="status" aria-live="polite">
      {notifications.map((toast) => (
        <article className={`toast-card ${toast.tone || "neutral"}`} key={toast.id}>
          <div>
            <strong>{toast.title}</strong>
            <span>{toast.message}</span>
          </div>
          <button type="button" onClick={() => onDismiss(toast.id)} aria-label="Dismiss notification">
            <X size={15} />
          </button>
        </article>
      ))}
    </div>
  );
}
