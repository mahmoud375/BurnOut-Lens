import { useEffect, useRef, memo } from "react";

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  radius: number;
  baseAlpha: number;
  phase: number;
}

export const AmbientBackground = memo(function AmbientBackground() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d", { alpha: true });
    if (!ctx) return;

    let animationFrameId: number;
    let width = (canvas.width = window.innerWidth);
    let height = (canvas.height = window.innerHeight);
    let isInteracting = false;

    // Pause particle updates during range slider interaction to eliminate thread contention
    const onInteractionStart = (e: Event) => {
      const target = e.target as HTMLElement | null;
      if (target && target.tagName === "INPUT" && (target as HTMLInputElement).type === "range") {
        isInteracting = true;
      }
    };
    const onInteractionEnd = () => {
      isInteracting = false;
    };

    window.addEventListener("pointerdown", onInteractionStart, { passive: true });
    window.addEventListener("pointerup", onInteractionEnd, { passive: true });
    window.addEventListener("pointercancel", onInteractionEnd, { passive: true });
    window.addEventListener("touchend", onInteractionEnd, { passive: true });
    window.addEventListener("mouseup", onInteractionEnd, { passive: true });

    // Calm, faint bioluminescent micro-particles (lightweight count: 28)
    const count = 28;
    const particles: Particle[] = Array.from({ length: count }, () => ({
      x: Math.random() * width,
      y: Math.random() * height,
      vx: (Math.random() - 0.5) * 0.2,
      vy: (Math.random() - 0.5) * 0.2,
      radius: Math.random() * 1.2 + 1.0,
      baseAlpha: Math.random() * 0.1 + 0.08,
      phase: Math.random() * Math.PI * 2,
    }));

    const handleResize = () => {
      if (!canvas) return;
      width = canvas.width = window.innerWidth;
      height = canvas.height = window.innerHeight;
    };

    window.addEventListener("resize", handleResize, { passive: true });

    let time = 0;
    const render = () => {
      animationFrameId = requestAnimationFrame(render);

      // Skip drawing while user is actively dragging a slider to guarantee 60fps interaction
      if (isInteracting) return;

      time += 0.01;
      ctx.clearRect(0, 0, width, height);

      for (let i = 0; i < particles.length; i++) {
        const p = particles[i];
        p.x += p.vx;
        p.y += p.vy;

        // Wrap-around screen bounds
        if (p.x < 0) p.x = width;
        else if (p.x > width) p.x = 0;
        if (p.y < 0) p.y = height;
        else if (p.y > height) p.y = 0;

        const alpha = Math.min(
          0.22,
          Math.max(0.04, p.baseAlpha + Math.sin(time + p.phase) * 0.04)
        );

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(206, 247, 158, ${alpha})`;
        ctx.fill();
      }
    };

    render();

    return () => {
      window.removeEventListener("resize", handleResize);
      window.removeEventListener("pointerdown", onInteractionStart);
      window.removeEventListener("pointerup", onInteractionEnd);
      window.removeEventListener("pointercancel", onInteractionEnd);
      window.removeEventListener("touchend", onInteractionEnd);
      window.removeEventListener("mouseup", onInteractionEnd);
      cancelAnimationFrame(animationFrameId);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        width: "100vw",
        height: "100vh",
        zIndex: -1,
        pointerEvents: "none",
        willChange: "transform",
        transform: "translateZ(0)", // Promotes to isolated GPU compositor layer
      }}
      aria-hidden="true"
    />
  );
});
