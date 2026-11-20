/**
 * electron-builder afterPack hook.
 *
 * macOS: ad-hoc sign the whole .app bundle (including the bundled Python
 * runtime).  Apple Silicon refuses to run arm64 code with no signature at all,
 * and an unsigned bundle shows the "app is damaged" dialog instead of the
 * normal "unidentified developer" prompt.  Ad-hoc signing needs no Apple
 * Developer account.  If a real Developer ID is configured (CSC_LINK), skip:
 * electron-builder signs and notarizes the bundle itself.
 */
const { execFileSync } = require('child_process');
const path = require('path');

exports.default = async function afterPack(context) {
  if (context.electronPlatformName !== 'darwin') return;
  if (process.env.CSC_LINK || process.env.CSC_NAME) return;

  const appName = `${context.packager.appInfo.productFilename}.app`;
  const appPath = path.join(context.appOutDir, appName);
  console.log(`  • ad-hoc signing ${appPath}`);
  execFileSync('codesign', ['--force', '--deep', '--sign', '-', appPath], { stdio: 'inherit' });
  execFileSync('codesign', ['--verify', '--deep', '--strict', appPath], { stdio: 'inherit' });
};
