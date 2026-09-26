import React from "react";

export default function LoadingSpinner({ label = "Loading..." }) {
  return <p className="muted">{label}</p>;
}
