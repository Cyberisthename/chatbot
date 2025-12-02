# Cortana Shell - Assets Attribution

This document details all assets extracted from the original Microsoft Cortana APK and how they're used in Cortana Shell 2.0.

## Source APK

- **Filename**: `cortana-3-2-0-12583-enus-release.apk`
- **Version**: 3.2.0.12583
- **Language**: English (US)
- **Publisher**: Microsoft Corporation
- **Original Platform**: Android
- **Package Name**: com.microsoft.cortana
- **Size**: ~58MB (58,152,089 bytes)

## Extracted Assets

### 1. Wake Word Detection Models

**Location**: `assets/wake-word-models/`

#### English (US) - Primary
- **File**: `heycortana_enUS.table`
- **Size**: 707,232 bytes (691 KB)
- **Source**: `extracted_bundles/cortana-3-2-0-12583-enus-release_apk/assets/heycortana_enUS.table`
- **Language**: en-US
- **Purpose**: Wake word detection for "Hey Cortana"
- **Usage**: Default model loaded by VoiceManager

#### English (US) - Alternative
- **File**: `heycortana_en-US.table`
- **Size**: 845,740 bytes (826 KB)
- **Source**: `extracted_bundles/cortana-3-2-0-12583-enus-release_apk/assets/heycortana_en-US.table`
- **Language**: en-US (alternative)
- **Purpose**: Alternative wake word model with different training
- **Usage**: Fallback or user-selectable alternative

#### English (India)
- **File**: `heycortana_enIN.table`
- **Size**: 85,504 bytes (83.5 KB)
- **Source**: `extracted_bundles/cortana-3-2-0-12583-enus-release_apk/assets/heycortana_enIN.table`
- **Language**: en-IN
- **Purpose**: Wake word detection for Indian English accent
- **Usage**: Multi-language support

#### Chinese (Simplified)
- **File**: `heycortana_zhCN.table`
- **Size**: 85,744 bytes (83.7 KB)
- **Source**: `extracted_bundles/cortana-3-2-0-12583-enus-release_apk/assets/heycortana_zhCN.table`
- **Language**: zh-CN
- **Purpose**: Wake word detection for Chinese speakers
- **Usage**: Multi-language support

**Technology**: Likely based on phoneme or acoustic models trained by Microsoft Research. Format is proprietary binary table format.

### 2. Icons & Branding

**Location**: `assets/icons/` and `assets/branding/`

#### Cortana Menu Icon
- **File**: `cortana.png`
- **Source**: `extracted_bundles/cortana-3-2-0-12583-enus-release_apk/res/drawable-xxhdpi-v4/cortana_icon_menu.png`
- **Resolution**: xxhdpi (extra-extra-high DPI)
- **Usage**: 
  - System tray icon
  - Window icon
  - Packaging icon (converted to .ico, .icns)
- **Description**: Classic Cortana circle icon with recognizable blue design

#### Cortana Double Ring Logo
- **File**: `cortanadouble.png`
- **Source**: `extracted_bundles/cortana-3-2-0-12583-enus-release_apk/res/drawable/cortanadouble.png`
- **Usage**: 
  - Splash screen (future)
  - About dialog
  - Branding materials
- **Description**: Double-ring Cortana logo, iconic brand element

### 3. UI Layouts (Reference Only)

These XML layouts were not directly ported but inspired the React UI design:

#### Voice Button Layout
- **Source**: `extracted_bundles/.../res/layout/voice_button.xml`
- **Inspiration**: Voice button design in Cortana Shell UI
- **Elements**: Button styling, positioning, interaction states

#### Cortana Profile Circle
- **Source**: `extracted_bundles/.../res/layout/cortana_profile_circle_layout.xml`
- **Inspiration**: Halo animation circle design
- **Elements**: Circle animations, sizing, positioning

#### Widget Information
- **Source**: `extracted_bundles/.../res/xml/cortana_widget_info.xml`
- **Inspiration**: Status bar and widget layout
- **Elements**: Compact UI, information density

### 4. Additional APK Resources (Not Extracted)

These assets exist in the APK but weren't used in current version:

#### Language Models
- `ial_en-us.db` - US English intent and action database
- `ial_en-gb.db` - UK English
- `ial_en-au.db` - Australian English
- `ial_en-ca.db` - Canadian English
- `ial_en-in.db` - Indian English
- `ial_fr-ca.db` - French Canadian
- `ial_zh-cn.db` - Simplified Chinese

**Potential Use**: Natural language understanding for commands

#### React Native Bundle
- `index.android.bundle` (4.2 MB)
- Original Cortana UI JavaScript bundle
- Not used (we built custom React UI from scratch)

#### Certificates & Security
- `cacert.pem` (265 KB)
- CA certificate bundle for HTTPS
- Not needed (using system certificates)

#### Context Templates
- `contextTemplate.json` (3 KB)
- Conversation context structure
- Potential future use for context management

#### Configuration
- `app_engine_client_config.xml`
- `ts_configuration.jwt`
- Original backend configuration
- Not applicable to custom backend

## Design Elements Inspired by Cortana

### Color Palette
- **Primary Blue**: #0078D7 (official Cortana blue)
- **Light Blue**: #00BCF2 (accent and highlights)
- **Accent Gold**: #FFB900 (notifications, warnings)
- **Background**: rgba(0, 0, 0, 0.7) (dark translucent)

### Animations
- **Halo Animation**: Pulsing concentric circles
  - Inspired by original Cortana listening animation
  - CSS keyframe animation with scale and opacity
  
- **Voice Pulse**: Red pulsing when listening
  - Mimics original microphone active state
  
- **Message Fade-In**: Smooth entry animations
  - Similar to original chat bubble animations

### Typography
- **Font**: Segoe UI (Windows system font)
- Matches original Cortana Windows 10 design

### Glassmorphism
- **Blur Effect**: backdrop-filter: blur(20px)
- **Translucency**: rgba backgrounds with alpha
- Inspired by Windows 10 Fluent Design System

## Legal & Ethical Use

### Copyright
- All original assets are © Microsoft Corporation
- Wake word models are proprietary Microsoft technology
- Icons and branding are Microsoft trademarks

### Usage Rights
This project uses Microsoft Cortana assets under the following principles:

1. **Educational Purpose**: Learning and demonstrating AI assistant technology
2. **Personal Use**: Non-commercial, individual use only
3. **Fair Use**: Limited use for interoperability and research
4. **Attribution**: Full credit to Microsoft Corporation
5. **No Distribution of Original Assets**: Models and icons not redistributed separately

### Restrictions
- ❌ Do not use for commercial purposes
- ❌ Do not redistribute Microsoft assets separately
- ❌ Do not claim Microsoft endorsement
- ❌ Do not violate Microsoft Cortana trademarks
- ✅ Give proper attribution to Microsoft
- ✅ Use for personal/educational projects
- ✅ Clearly indicate this is a fan/hobby project

### Disclaimer
**Cortana Shell 2.0 is an independent project and is not affiliated with, endorsed by, or connected to Microsoft Corporation. "Cortana" is a trademark of Microsoft Corporation. All Microsoft assets are used respectfully and for educational purposes only.**

## Technical Details

### Wake Word Model Format

The `.table` files are binary models with the following characteristics:

- **Format**: Proprietary binary table
- **Encoding**: Little-endian
- **Structure**: Header + phoneme mappings + acoustic features
- **Detection**: Likely uses Hidden Markov Models (HMM) or neural networks
- **Threshold**: Adjustable sensitivity parameter

**Current Usage**: Loaded into memory by VoiceManager but not yet fully integrated (requires compatible detection library).

### Icon Formats

- **PNG** (current): Used for all platforms
- **ICO** (needed): Windows executable icon (conversion required)
- **ICNS** (needed): macOS app bundle icon (conversion required)
- **SVG** (future): Scalable vector version for perfect rendering

**Conversion Commands**:
```bash
# PNG to ICO (Windows)
convert cortana.png -define icon:auto-resize=256,128,64,48,32,16 cortana.ico

# PNG to ICNS (macOS)
mkdir cortana.iconset
sips -z 16 16   cortana.png --out cortana.iconset/icon_16x16.png
sips -z 32 32   cortana.png --out cortana.iconset/icon_16x16@2x.png
# ... (more sizes)
iconutil -c icns cortana.iconset
```

## Future Asset Improvements

### Planned Additions
- [ ] Convert PNG icons to ICO/ICNS for proper packaging
- [ ] Extract and use additional language models
- [ ] Port more UI animations from APK
- [ ] Create custom Cortana-inspired SVG icons
- [ ] Implement full wake word detection pipeline
- [ ] Add sound effects from APK (if available)

### Alternative Assets
If Microsoft assets cannot be used:
- Create original wake word models (Mycroft, Porcupine)
- Design original icons inspired by Cortana aesthetic
- Use open-source alternatives for all proprietary components

## Acknowledgments

Special thanks to:
- **Microsoft Corporation** for creating the original Cortana
- **Microsoft Research** for wake word detection technology
- **Cortana Design Team** for the iconic blue halo design
- **Microsoft** for Fluent Design System inspiration

## References

- Microsoft Cortana: https://www.microsoft.com/cortana
- Fluent Design System: https://www.microsoft.com/design/fluent
- Windows UI Guidelines: https://docs.microsoft.com/windows/apps/design/

---

**All assets used respectfully and with full attribution to Microsoft Corporation.** 💙
