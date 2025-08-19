### From https://github.com/Marktechpost/AI-Tutorial-Codes-Included/blob/main/cipher_memory_agent_Marktechpost.ipynb

import os, getpass
os.environ["GEMINI_API_KEY"] = getpass.getpass("Enter your Gemini API key: ").strip()

import subprocess, tempfile, pathlib, textwrap, time, requests, shlex

def choose_llm():
    if os.getenv("OPENAI_API_KEY"):
        return "openai", "gpt-4o-mini", "OPENAI_API_KEY"
    if os.getenv("GEMINI_API_KEY"):
        return "gemini", "gemini-2.5-flash", "GEMINI_API_KEY"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic", "claude-3-5-haiku-20241022", "ANTHROPIC_API_KEY"
    raise RuntimeError("Set one API key before running.")

def run(cmd, check=True, env=None):
    print("▸", cmd)
    p = subprocess.run(cmd, shell=True, text=True, capture_output=True, env=env)
    if p.stdout: print(p.stdout)
    if p.stderr: print(p.stderr)
    if check and p.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")
    return p

def ensure_node_and_cipher():
    run("sudo apt-get update -y && sudo apt-get install -y nodejs npm", check=False)
    run("npm install -g @byterover/cipher")
     
def write_cipher_yml(workdir, provider, model, key_env):
    cfg = """
llm:
  provider: {provider}
  model: {model}
  apiKey: ${key_env}
systemPrompt:
  enabled: true
  content: |
    You are an AI programming assistant with long-term memory of prior decisions.
embedding:
  disabled: true
mcpServers:
  filesystem:
    type: stdio
    command: npx
    args: ['-y','@modelcontextprotocol/server-filesystem','.']
""".format(provider=provider, model=model, key_env=key_env)

    (workdir / "memAgent").mkdir(parents=True, exist_ok=True)
    (workdir / "memAgent" / "cipher.yml").write_text(cfg.strip() + "\n")

def cipher_once(text, env=None, cwd=None):
    cmd = f'cipher {shlex.quote(text)}'
    p = subprocess.run(cmd, shell=True, text=True, capture_output=True, env=env, cwd=cwd)
    print("Cipher says:\n", p.stdout or p.stderr)
    return p.stdout.strip() or p.stderr.strip()

def start_api(env, cwd):
    proc = subprocess.Popen("cipher --mode api", shell=True, env=env, cwd=cwd,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for _ in range(30):
        try:
            r = requests.get("http://127.0.0.1:3000/health", timeout=2)
            if r.ok:
                print("API /health:", r.text)
                break
        except: pass
        time.sleep(1)
    return proc

def main():
    provider, model, key_env = choose_llm()
    ensure_node_and_cipher()
    workdir = pathlib.Path(tempfile.mkdtemp(prefix="cipher_demo_"))
    write_cipher_yml(workdir, provider, model, key_env)
    env = os.environ.copy()

    cipher_once("Store decision: use pydantic for config validation; pytest fixtures for testing.", env, str(workdir))
    cipher_once("Remember: follow conventional commits; enforce black + isort in CI.", env, str(workdir))

    cipher_once("What did we standardize for config validation and Python formatting?", env, str(workdir))

    api_proc = start_api(env, str(workdir))
    time.sleep(3)
    api_proc.terminate()

if __name__ == "__main__":
    main()
     
