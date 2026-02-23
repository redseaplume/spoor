// Static file server for Playwright tests — serves web/ on port 3111
const server = Bun.serve({
  port: 3111,
  async fetch(req) {
    const url = new URL(req.url);
    let path = url.pathname === "/" ? "/index.html" : url.pathname;

    const file = Bun.file(`web${path}`);
    if (await file.exists()) {
      return new Response(file);
    }

    return new Response("Not found", { status: 404 });
  },
});

console.log(`[test server] serving web/ on http://localhost:${server.port}`);
