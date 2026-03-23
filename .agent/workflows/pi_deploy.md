---
description: Deploy the hangermon code base to a remote Raspberry Pi
---

# Deploy to Raspberry Pi

Follow these steps to push the `hangermon` codebase from your local machine to the target Raspberry Pi.

1. Ensure the Raspberry Pi is reachable via SSH.
2. Run the deployment script with the PI_HOST and PI_USER:

// turbo
3. Push to Pi and start the service:
```bash
./scripts/deploy_to_pi.sh [PI_HOST] [PI_USER]
```

4. Once the deployment script completes, you can view the dashboard at `http://[PI_HOST]:8000`.
