# Oracle Cloud Always Free deployment (ARM64)

This profile is designed for the current Oracle Cloud Always Free Ampere A1 allowance. It keeps the Flask application, BGE-M3 dense embeddings, FAISS, Groq integration, SQLite chat history and uploaded files. Because PaddlePaddle's official Linux CPU support currently targets x86-64, this ARM64 profile uses Tesseract OCR for scanned PDFs instead of Indra/PaddleOCR.

## What is included

- Docker image for Ubuntu/OCI Ampere A1 ARM64.
- Gunicorn with one worker and four request threads.
- BGE-M3 dense embeddings on CPU (batch size 1).
- Groq API for LLM inference.
- FAISS and SQLite stored on a persistent Docker volume.
- Tesseract OCR (English language data included).
- Separate storage directories for each browser session. Visitors do not share uploaded documents, FAISS indexes or chat databases.
- A 20 MB per-file upload limit and 250 MB uploaded-file quota per browser workspace.
- A cleanup script for inactive workspaces (default retention: 14 days).
- Caddy reverse proxy for automatic HTTPS.

## Free-tier limits to plan against

Oracle's current Always Free A1 allowance is **2 OCPUs and 12 GB RAM per tenancy**, not the older 4 OCPU / 24 GB figures in older tutorials. The 200 GB Block Volume allowance is shared by boot and attached volumes. A1 is ARM-based and capacity is subject to availability in the account's home region. Oracle can reclaim instances it considers idle. See:
- https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm
- https://www.oracle.com/cloud/free/

PaddlePaddle's current Linux CPU installation documentation states x86-64 processor support; this deployment therefore does not install PaddleOCR/Indra OCR. Tesseract is used instead. The default Tesseract language is English; additional Tesseract language packages and a matching OCR_LANG value are needed for other languages.

## 1. Create the Oracle VM

1. Create an Oracle Cloud Free Tier account and choose your home region carefully. Oracle asks for payment-card details for identity verification; Always Free resources are not the same as the temporary trial credits.
2. In the OCI Console, create a VM instance using **VM.Standard.A1.Flex** and an Ubuntu ARM64 image.
3. Allocate no more than **2 OCPUs and 12 GB memory** total to A1 instances in the tenancy.
4. Use the default boot volume (or another size within the shared 200 GB Always Free Block Volume allowance).
5. Assign a public IPv4 address and configure the subnet security list / network security group to allow inbound TCP **22, 80 and 443**. Keep port 8000 private.
6. Connect over SSH.

If OCI reports out-of-host-capacity, retry a different availability domain (where available) or try again later. Do not upgrade to a paid account solely to get around capacity unless you explicitly accept possible charges.

## 2. Install Docker on Ubuntu ARM64

Follow Docker's official Ubuntu installation instructions for the ARM64 architecture:
https://docs.docker.com/engine/install/ubuntu/

Install the Docker Compose plugin and verify with these commands:

~~~bash
docker --version
docker compose version
~~~

Add your SSH user to the docker group only if you understand that docker-group membership grants root-equivalent privileges.

## 3. Configure DNS

Create a free DNS hostname (for example, a DuckDNS subdomain) and point its A record to the VM's public IPv4 address. Wait for DNS to resolve before starting Caddy. Caddy will request and renew a Let's Encrypt HTTPS certificate automatically.

## 4. Deploy the repository

~~~bash
git clone --branch deploy/oracle-always-free https://github.com/Swaraj-Palai/Multi-Document_AI_Information_Extraction.git
cd Multi-Document_AI_Information_Extraction
cp deploy/oracle/.env.example deploy/oracle/.env
~~~

Edit deploy/oracle/.env and set:
- APP_SECRET_KEY: generate with openssl rand -hex 32.
- GROQ_API_KEY: your Groq API key.
- PUBLIC_HOSTNAME: the DNS hostname pointing to the VM.
- ACME_EMAIL: an email for certificate expiry notifications.

Never commit the filled .env file or put secrets in GitHub.

Start the app:

~~~bash
docker compose -f deploy/oracle/compose.yaml up -d --build
docker compose -f deploy/oracle/compose.yaml logs -f app
~~~

The model downloads on first start and is cached on the persistent app_data volume. First startup can take several minutes. Check logs for successful BGE-M3 initialization and for Tesseract availability.

## 5. Check the deployment

Open:

~~~text
https://YOUR_PUBLIC_HOSTNAME
~~~

The app listens internally on port 8000; Caddy publishes only HTTPS (443) and HTTP redirect / certificate challenge traffic (80).

## 6. Workspace cleanup

Each browser gets isolated documents, vector index and chats. Inactive workspaces should be pruned to control disk consumption. Run daily from cron:

~~~cron
17 3 * * * cd /path/to/Multi-Document_AI_Information_Extraction && docker compose -f deploy/oracle/compose.yaml exec -T app python scripts/cleanup_expired_workspaces.py --days 14 >> /var/log/mdai-workspace-cleanup.log 2>&1
~~~

The container needs the same persistent /data volume. Adjust the checkout path and retention to suit.

## Notes and limitations

- This is a free-tier demo target, not a production hosting SLA. Two OCPUs mean model loading, OCR and embedding jobs may be slow under concurrent use.
- The BGE-M3 model runs on CPU. The first model download uses disk and memory; subsequent starts reuse the cache.
- The OCR replacement in this profile uses Tesseract in English by default. It is not feature-equivalent to Indra/PaddleOCR for all supported languages or layouts.
- No visitor account system is included; the Flask signed session cookie isolates browser workspaces. Users should still avoid uploading confidential documents to a public demo.
- Anonymous public uploads can be abused. The app enforces per-file and per-workspace size limits; monitor disk use and keep the workspace cleanup schedule active.
- Back up the Docker app_data volume before operating-system maintenance or VM replacement.
- Do not expose the app's internal port 8000 directly to the internet.
