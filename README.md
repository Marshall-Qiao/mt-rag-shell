# mt-rag-shell

可独立使用的金融问答 RAG 系统骨架：FastAPI + WebSocket、LangGraph、Chroma、React，
支持可选的 LangSmith 观测。保留问句改写、并行检索和时效评分、相关性判断及回答生成。

此副本只包含源码、依赖锁文件、空配置模板和使用虚构文档的离线测试。
不包含原始文档、向量数据库、聊天记录、凭据、运行缓存或原仓库 Git 历史。
两个示例集合使用通用名称 `rag-default` 和 `rag-reference`。

## 环境与离线验证

使用 Python 3.11 或 3.12。前端可选。

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
```

Windows 将 `.venv/bin/python` 替换为 `.venv/Scripts/python.exe`。
测试使用虚构内容、模拟模型和内存存储，禁止网络连接，不需要真实 API key 或向量库。
已有依赖环境可以直接运行测试，不需要为本副本再复制一份环境。

## 启动系统

```sh
cp .env.example .env
.venv/bin/python -m uvicorn chat-server:app --host 127.0.0.1 --port 8080
```

生产安装只需 `requirements.txt`。服务导入和启动不会打开向量库或调用模型。
`GET /health/live` 返回进程存活状态，`GET /connect` 用于前端连接检查。
这两个接口不证明数据库或模型可用。

实际问答需要在 `.env` 中填写 `OPENAI_API_KEY`。问答和文档导入会调用外部模型或嵌入 API，
会向供应商发送对应问题、上下文或文档内容，并可能计费。请在授权相应数据与费用范围后执行。
发布源码不需要真实 key，也不需要导入任何文档。

模型按角色通过 `ENTRY_MODEL`、`RELEVANCE_MODEL`、`TIMELINESS_MODEL`、`ANSWER_MODEL` 配置。
`EMBEDDING_MODEL` 必须与已有向量一致；修改它需要重新建库。
进程环境变量优先于 `.env`，`MT_RAG_ENV_FILE` 可指定其他配置文件。

## 导入自己的资料

文档和索引统一放在已被 Git 忽略的 `data/` 下。没有自动入库或示例业务数据。

```sh
mkdir -p data/documents/default data/documents/reference
# 将自己的 PDF 放入 data/documents/default 后，明确执行导入：
.venv/bin/python scripts/ingest.py data/documents/default --format pdf --collection default
# 或导入自己的文本资料：
.venv/bin/python scripts/ingest.py data/documents/reference --format txt --collection reference
```

默认索引目录为 `data/vector-store`，可通过 `VECTOR_DB_PATH` 指定。
两个集合名通过 `DEFAULT_COLLECTION` 和 `REFERENCE_COLLECTION` 配置。
空目录会明确报错，不会创建数据库或发送嵌入请求。

未变化的文件按相对路径和修改时间跳过导入；不同子目录的同名文件不会混淆。
修改后的文件目前会追加新版本，**不会自动删除旧版本**；大规模更新前应设计替换或重建策略。
时效字段默认使用文件修改时间，它不代表文档真实发布日期；正式资料应补充可信发布日期。
不要直接让本副本打开旧版 Chroma 数据目录，先备份并验证格式和嵌入模型兼容性。

## 可选前端

```sh
cd demo/front-end
npm ci
npm start
```

默认访问 `http://localhost:3000`，连接 `http://localhost:8080`。
`demo/front-end/.env.example` 提供 `REACT_APP_API_URL` 和 `REACT_APP_WS_URL` 模板；
自定义时复制为该目录的 `.env` 后重启前端。跨域来源由后端 `CORS_ORIGINS` JSON 数组控制。
前端已移除不生效的模型切换控件。`POST /chain` 仅保留原兼容性应答，不会更改运行模型。

当前服务适用于本机开发，没有用户认证、限流或配额管理。
WebSocket 校验浏览器 Origin，但不替代认证。对外部署前需补齐访问控制、TLS、超时和额度限制。
每个 WebSocket 会话使用独立的内存历史，断开后删除，重启后丢失；不支持跨进程共享。
模型返回节点结果后发送完整消息，目前不提供逐 token 输出。

## 观测与后续优化

`LANGSMITH_TRACING=false` 为默认值，默认隐藏输入和输出。
开启观测会向外部持久存储发送请求标识、时延和调用信息，可能计费；
关闭隐藏开关还会上传问题、检索正文和回答。仅在授权范围内启用。
`scripts/verify_langsmith.py` 是在线验证工具，不属于离线测试，本次抽取未执行它。
`AIAgentPoC/agents/*/test.py` 是会调用真实模型和远程图形渲染的交互演示，也不属于离线测试。

已处理的具体问题：旧入库函数重复提交文档、去重时间字段跨平台不兼容、
子目录同名文件冲突、两组检索串行、内部 IP 硬编码，以及把本地数据装进 Git/Docker 上下文的风险。

仍可优化的方向：两组集合共享一次问题嵌入、导入时只解析新增文件、
更新文档时替换旧分块、检索前应用时效过滤、可配置的上下文长度、会话历史上限和请求超时。
应使用可信样本集评估召回、引用与回答质量后再调整评分或模型，不能以离线模拟测试替代质量验收。

## 发布与数据边界

该骨架使用独立的新 Git 历史，不继承原项目的提交记录。
`.env.example` 中 key 留空；`.gitignore` 和 `.dockerignore` 排除数据、索引、凭据及环境。
新增文件时仍需检查公开范围。仓库目前未指定开源许可证。
