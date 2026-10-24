# 新生百科 · 部署仓库

一个仓库同时部署：

- **Streamlit 在线体验**：`src/05_app.py`
- **H5 展示页**：`docs/index.html`（GitHub Pages 选 `/docs`）

## 1. 上传到 GitHub

1. 新建一个 **Public** 仓库（例如 `xinsheng-baike`）。
2. 在空仓库页面点 **uploading an existing file**。
3. 把本文件夹（`deploy_all`）里的**内容**拖进去（不是把 `deploy_all` 文件夹本身拖进去）：
   - `src/`
   - `data/`
   - `docs/`
   - `requirements.txt`
   - `runtime.txt`
   - `.env.example`
   - `.gitignore`
   - `README.md`
4. Commit。

> **不要上传 `.env`**。API Key 只放在 Streamlit Cloud 的 Secrets 里。

## 2. 部署 Streamlit

1. 打开 https://share.streamlit.io/ ，用 GitHub 登录并授权。
2. **New app** → 选择你的仓库 / `main` 分支 / 主文件路径填：

   ```
   src/05_app.py
   ```

3. 点 **Advanced settings → Secrets**，填：

   ```toml
   DEEPSEEK_API_KEY = "sk-你的key"
   ```

4. Deploy，等 2–5 分钟。
5. 得到一个 `https://xxx.streamlit.app` 地址；打开、问一个问题确认能用。

## 3. 部署 H5

1. GitHub 仓库 → **Settings → Pages**。
2. Source 选 **Deploy from a branch**。
3. Branch 选 **main**，目录选 **`/docs`**。
4. Save，等 1–2 分钟。
5. 得到 `https://<用户名>.github.io/<仓库名>/`。

## 4. 填 H5 里的三个链接

编辑 `docs/index.html`，搜索 `const CONFIG`，填：

```js
const CONFIG = {
  appUrl: "https://xxx.streamlit.app",
  videoUrl: "video/demo.mp4",
  repoUrl: "https://github.com/<用户名>/<仓库名>",
};
```

- 视频放到 `docs/video/demo.mp4`，或在 `videoUrl` 填外部视频链接。
- 保存后等待 GitHub Pages 自动更新（约 1 分钟）。

## 5. 提交

- 作品链接填 **H5 的 GitHub Pages 地址**。
- 评委打开 H5 → 点“立即体验” → 进入 Streamlit。
