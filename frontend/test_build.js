const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

try {
  console.log("Building...");
  execSync('npm run build', { cwd: 'C:\\MISCSSSS 2.0\\AREA 51\\rag\\data injestion pipeline\\frontend', stdio: 'pipe' });
  
  console.log("Build complete. Checking dist/assets...");
  const assetsDir = 'C:\\MISCSSSS 2.0\\AREA 51\\rag\\data injestion pipeline\\frontend\\dist\\assets';
  const files = fs.readdirSync(assetsDir);
  
  let found = false;
  for (const file of files) {
    if (file.endsWith('.js')) {
      const content = fs.readFileSync(path.join(assetsDir, file), 'utf8');
      if (content.includes('posthog')) {
        console.log(`FOUND posthog in ${file}`);
        found = true;
      }
    }
  }
  if (!found) console.log("PostHog NOT found in any JS file.");
} catch (e) {
  console.error("Error:", e.stdout ? e.stdout.toString() : e.message);
}
