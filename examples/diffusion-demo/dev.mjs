// Dev launcher: run the API server and the Vite dev server together.
import { spawn } from "node:child_process";

function run(cmd, args, name) {
  const p = spawn(cmd, args, { stdio: "pipe", shell: process.platform === "win32" });
  const tag = `[${name}] `;
  p.stdout.on("data", (d) => process.stdout.write(tag + d.toString().replace(/\n(?=.)/g, "\n" + tag)));
  p.stderr.on("data", (d) => process.stderr.write(tag + d.toString().replace(/\n(?=.)/g, "\n" + tag)));
  p.on("exit", (code) => {
    console.log(`${tag}exited with ${code}`);
    process.exit(code ?? 0);
  });
  return p;
}

run("node", ["server/index.mjs"], "api");
const npx = process.platform === "win32" ? "npx.cmd" : "npx";
run(npx, ["vite"], "web");
