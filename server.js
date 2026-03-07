/**
 * ╔══════════════════════════════════════════════════════════╗
 * ║   AVRIL.MA — Skin Analyzer Backend                       ║
 * ║   Serveur Node.js qui reçoit une photo de peau,          ║
 * ║   l'envoie à GPT-4o Vision et retourne l'analyse.        ║
 * ╚══════════════════════════════════════════════════════════╝
 */

require('dotenv').config();
const express    = require('express');
const cors       = require('cors');
const multer     = require('multer');
const OpenAI     = require('openai');
const rateLimit  = require('express-rate-limit');

const app  = express();
const port = process.env.PORT || 3001;

// ── OpenAI client ─────────────────────────────────────────
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// ── CORS : n'accepter que les requêtes depuis avril.ma ────
const allowedOrigins = [
  process.env.SHOPIFY_STORE_URL || 'https://www.avril.ma',
  'http://localhost:3000',
  'http://127.0.0.1:5500',   // live-server local pour tests
];
app.use(cors({
  origin: (origin, callback) => {
    if (!origin || allowedOrigins.some(o => origin.startsWith(o))) {
      callback(null, true);
    } else {
      callback(new Error('CORS bloqué : origine non autorisée'));
    }
  }
}));

// ── Rate limiting : max 20 analyses/minute par IP ────────
const limiter = rateLimit({
  windowMs: 60 * 1000,
  max: 20,
  message: { error: 'Trop de requêtes. Veuillez patienter une minute.' }
});
app.use('/api/', limiter);

// ── Multer : upload en mémoire, max 8 Mo ─────────────────
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 8 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    const allowed = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
    if (allowed.includes(file.mimetype)) cb(null, true);
    else cb(new Error('Format non supporté. Utilisez JPG, PNG ou WEBP.'));
  }
});

// ── Prompt envoyé à GPT-4o Vision ────────────────────────
const SKIN_ANALYSIS_PROMPT = `
Tu es un expert dermatologue et conseiller skincare. Analyse cette photo de visage humain.

Évalue les 4 dimensions suivantes et retourne UNIQUEMENT un objet JSON valide, sans texte avant ni après :

{
  "acne": <score 0-100, 0=peau nette, 100=acné sévère>,
  "hydration": <score 0-100, 0=très déshydratée, 100=parfaitement hydratée>,
  "radiance": <score 0-100, 0=teint très terne, 100=teint très lumineux>,
  "firmness": <score 0-100, 0=très relâchée, 100=très ferme>,

  "acne_detail": "<phrase de 1-2 lignes en français décrivant l'état de l'acné et des pores>",
  "hydration_detail": "<phrase de 1-2 lignes en français décrivant le niveau d'hydratation>",
  "radiance_detail": "<phrase de 1-2 lignes en français décrivant le teint, les taches, l'éclat>",
  "firmness_detail": "<phrase de 1-2 lignes en français décrivant la fermeté et les signes de vieillissement>",

  "skin_type": "<'grasse' | 'sèche' | 'mixte' | 'normale'>",
  "global_score": <score global 0-100>,
  "priority_concern": "<le problème principal à traiter en priorité, en 5 mots max>"
}

Si la photo ne montre pas clairement un visage humain, retourne :
{"error": "Veuillez télécharger une photo claire de votre visage."}
`.trim();

// ══════════════════════════════════════════════════════════
// ROUTE PRINCIPALE : POST /api/analyze
// ══════════════════════════════════════════════════════════
app.post('/api/analyze', upload.single('photo'), async (req, res) => {
  try {
    // Vérification : photo présente ?
    if (!req.file) {
      return res.status(400).json({ error: 'Aucune photo reçue.' });
    }

    // Convertir en base64 pour l'API OpenAI
    const base64Image = req.file.buffer.toString('base64');
    const mimeType    = req.file.mimetype;

    console.log(`[${new Date().toISOString()}] Analyse en cours — ${req.file.originalname} (${(req.file.size/1024).toFixed(0)} Ko)`);

    // ── Appel GPT-4o Vision ──────────────────────────────
    const response = await openai.chat.completions.create({
      model: 'gpt-4o',
      max_tokens: 600,
      messages: [
        {
          role: 'user',
          content: [
            {
              type: 'image_url',
              image_url: {
                url: `data:${mimeType};base64,${base64Image}`,
                detail: 'high'   // haute résolution pour meilleure analyse
              }
            },
            {
              type: 'text',
              text: SKIN_ANALYSIS_PROMPT
            }
          ]
        }
      ]
    });

    // ── Parser la réponse JSON ───────────────────────────
    const rawText = response.choices[0].message.content.trim();

    let analysis;
    try {
      // Extraire le JSON même si GPT ajoute du texte autour
      const jsonMatch = rawText.match(/\{[\s\S]*\}/);
      if (!jsonMatch) throw new Error('JSON introuvable dans la réponse');
      analysis = JSON.parse(jsonMatch[0]);
    } catch (parseErr) {
      console.error('Erreur parsing JSON GPT:', rawText);
      return res.status(500).json({ error: 'Erreur lors de l\'analyse. Réessayez.' });
    }

    // ── Si GPT détecte que ce n'est pas un visage ────────
    if (analysis.error) {
      return res.status(422).json({ error: analysis.error });
    }

    // ── Construire les recommandations de produits ───────
    const recommendations = buildRecommendations(analysis);

    // ── Réponse finale ───────────────────────────────────
    const result = {
      success: true,
      analysis,
      recommendations,
      meta: {
        model: 'gpt-4o',
        analyzed_at: new Date().toISOString(),
        cost_estimate_eur: '~0.01'
      }
    };

    console.log(`[OK] global_score=${analysis.global_score}, skin_type=${analysis.skin_type}`);
    return res.json(result);

  } catch (err) {
    console.error('Erreur serveur:', err.message);
    if (err.code === 'insufficient_quota') {
      return res.status(503).json({ error: 'Quota OpenAI dépassé. Contactez l\'administrateur.' });
    }
    return res.status(500).json({ error: 'Erreur serveur. Réessayez dans quelques instants.' });
  }
});

// ══════════════════════════════════════════════════════════
// LOGIQUE DE RECOMMANDATION PRODUITS
// ── Adapte ces noms/URLs à votre vrai catalogue avril.ma ─
// ══════════════════════════════════════════════════════════
function buildRecommendations(analysis) {
  const { acne, hydration, radiance, firmness, skin_type } = analysis;
  const recs = [];

  // ── Acné & Imperfections ─────────────────────────────
  if (acne >= 40) {
    recs.push({
      category: 'Acné',
      icon: '🔴',
      name: 'Sérum Anti-Imperfections Zinc & Niacinamide',
      reason: 'Réduit les pores et régule le sébum',
      urgency: acne >= 65 ? 'urgent' : 'recommandé',
      product_url: 'https://www.avril.ma/collections/soins-visage/products/serum-anti-imperfections',
      usage: 'Matin & soir après nettoyage'
    });
  }

  // ── Nettoyant selon type de peau ─────────────────────
  if (skin_type === 'grasse' || acne >= 30) {
    recs.push({
      category: 'Nettoyage',
      icon: '🫧',
      name: 'Gel Nettoyant Purifiant Acide Salicylique',
      reason: 'Nettoie en profondeur sans dessécher',
      urgency: 'recommandé',
      product_url: 'https://www.avril.ma/collections/nettoyants/products/gel-nettoyant-purifiant',
      usage: 'Matin & soir, 60 secondes de massage'
    });
  } else {
    recs.push({
      category: 'Nettoyage',
      icon: '🌿',
      name: 'Lait Nettoyant Douceur Aloe Vera',
      reason: 'Nettoyage doux pour peau normale/sèche',
      urgency: 'recommandé',
      product_url: 'https://www.avril.ma/collections/nettoyants/products/lait-nettoyant-douceur',
      usage: 'Matin & soir'
    });
  }

  // ── Hydratation ──────────────────────────────────────
  if (hydration < 60) {
    recs.push({
      category: 'Hydratation',
      icon: '💧',
      name: 'Sérum Acide Hyaluronique Triple Action',
      reason: 'Hydratation intense sur 3 niveaux de profondeur',
      urgency: hydration < 40 ? 'urgent' : 'recommandé',
      product_url: 'https://www.avril.ma/collections/serums/products/serum-acide-hyaluronique',
      usage: 'Matin & soir sur peau humide, avant crème'
    });
  }

  // ── Crème hydratante ─────────────────────────────────
  if (skin_type === 'sèche' || hydration < 50) {
    recs.push({
      category: 'Soin Quotidien',
      icon: '✨',
      name: 'Crème Hydratante Riche Karité & Ceramides',
      reason: 'Nourrit et répare la barrière cutanée',
      urgency: 'recommandé',
      product_url: 'https://www.avril.ma/collections/cremes/products/creme-hydratante-riche',
      usage: 'Matin & soir'
    });
  } else {
    recs.push({
      category: 'Soin Quotidien',
      icon: '🌸',
      name: 'Fluide Hydratant Léger SPF 30',
      reason: 'Hydratation légère avec protection solaire',
      urgency: 'recommandé',
      product_url: 'https://www.avril.ma/collections/cremes/products/fluide-hydratant-spf30',
      usage: 'Chaque matin'
    });
  }

  // ── Éclat & Taches ───────────────────────────────────
  if (radiance < 65) {
    recs.push({
      category: 'Éclat',
      icon: '☀️',
      name: 'Sérum Vitamine C Stabilisée 15%',
      reason: 'Unifie le teint et illumine en 4 semaines',
      urgency: radiance < 45 ? 'urgent' : 'recommandé',
      product_url: 'https://www.avril.ma/collections/serums/products/serum-vitamine-c',
      usage: 'Chaque matin avant la crème'
    });
  }

  // ── Anti-âge / Fermeté ───────────────────────────────
  if (firmness < 70) {
    recs.push({
      category: 'Anti-Âge',
      icon: '⏳',
      name: 'Sérum Rétinol Nuit 0.3% + Peptides',
      reason: 'Stimule le renouvellement cellulaire et le collagène',
      urgency: firmness < 50 ? 'urgent' : 'recommandé',
      product_url: 'https://www.avril.ma/collections/serums/products/serum-retinol-nuit',
      usage: 'Soir uniquement, 2-3x par semaine au début'
    });
  }

  // ── SPF (toujours recommandé) ─────────────────────────
  recs.push({
    category: 'Protection',
    icon: '🛡️',
    name: 'Écran Solaire SPF 50+ Invisible',
    reason: 'Protection essentielle contre les UV (anti-taches, anti-âge)',
    urgency: 'essentiel',
    product_url: 'https://www.avril.ma/collections/solaires/products/ecran-solaire-spf50',
    usage: 'Chaque matin en dernière étape, à renouveler'
  });

  // Retourner les 4 recommandations les plus pertinentes
  const order = { urgent: 0, essentiel: 1, recommandé: 2 };
  return recs
    .sort((a, b) => order[a.urgency] - order[b.urgency])
    .slice(0, 4);
}


// ══════════════════════════════════════════════════════════
// THEME UPLOAD SETUP : GET /setup?code=OAUTH_CODE
// ══════════════════════════════════════════════════════════
const https = require('https');

const SHOPIFY_SHOP    = 'avril-beauty.myshopify.com';
const SHOPIFY_THEME_ID = '191446909275';
const CLIENT_ID       = 'be91427fd405d0c288feab04b8026fe2';
const CLIENT_SECRET   = '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';

function shopifyPost(path, body) {
  return new Promise((resolve, reject) => {
    const data = JSON.stringify(body);
    const req = https.request({
      hostname: SHOPIFY_SHOP, path, method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(data) }
    }, (res) => {
      let d = '';
      res.on('data', c => d += c);
      res.on('end', () => resolve({ status: res.statusCode, body: d }));
    });
    req.on('error', reject); req.write(data); req.end();
  });
}

function shopifyPut(path, body, token) {
  return new Promise((resolve, reject) => {
    const data = JSON.stringify(body);
    const req = https.request({
      hostname: SHOPIFY_SHOP, path, method: 'PUT',
      headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(data), 'X-Shopify-Access-Token': token }
    }, (res) => {
      let d = '';
      res.on('data', c => d += c);
      res.on('end', () => resolve({ status: res.statusCode, body: d }));
    });
    req.on('error', reject); req.write(data); req.end();
  });
}

const THEME_FILES = {
  'sections/skin-analyzer.liquid': `{% comment %}
  Avril.ma - AI Skin Analyzer Section
  Backend: https://web-production-fb65c.up.railway.app
{% endcomment %}

<link rel="stylesheet" href="{{ 'skin-analyzer.css' | asset_url }}">

<section class="avril-skin-analyzer" id="avril-skin-analyzer">
  <div class="analyzer-container">

    <div class="analyzer-header">
      <h2 class="analyzer-title">{{ section.settings.title }}</h2>
      <p class="analyzer-subtitle">{{ section.settings.subtitle }}</p>
    </div>

    <div class="analyzer-steps">
      <!-- Step 1: Upload -->
      <div class="step step-upload" id="step-upload">
        <div class="upload-area" id="upload-area">
          <div class="upload-icon">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
              <polyline points="17 8 12 3 7 8"/>
              <line x1="12" y1="3" x2="12" y2="15"/>
            </svg>
          </div>
          <p class="upload-text">{{ section.settings.upload_text }}</p>
          <p class="upload-hint">JPG, PNG — max 5 MB</p>
          <input type="file" id="skin-photo-input" accept="image/jpeg,image/png,image/webp" hidden>
          <button class="btn-upload" onclick="document.getElementById('skin-photo-input').click()">
            {{ section.settings.upload_btn_text }}
          </button>
        </div>

        <div class="photo-preview hidden" id="photo-preview">
          <img id="preview-img" src="" alt="Votre photo">
          <button class="btn-change-photo" onclick="resetAnalyzer()">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/>
              <path d="M3 3v5h5"/>
            </svg>
            Changer la photo
          </button>
        </div>

        <button class="btn-analyze hidden" id="btn-analyze" onclick="analyzePhoto()">
          {{ section.settings.analyze_btn_text }}
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
          </svg>
        </button>
      </div>

      <!-- Step 2: Loading -->
      <div class="step step-loading hidden" id="step-loading">
        <div class="loading-animation">
          <div class="loading-dots">
            <span></span><span></span><span></span>
          </div>
          <p class="loading-text">{{ section.settings.loading_text }}</p>
        </div>
      </div>

      <!-- Step 3: Results -->
      <div class="step step-results hidden" id="step-results">
        <div class="results-header">
          <h3>Votre analyse personnalisée</h3>
        </div>

        <div class="results-grid">
          <div class="result-card" id="result-skin-type">
            <div class="result-icon">🧬</div>
            <h4>Type de peau</h4>
            <p id="skin-type-text">—</p>
          </div>
          <div class="result-card" id="result-concerns">
            <div class="result-icon">🔍</div>
            <h4>Préoccupations</h4>
            <p id="concerns-text">—</p>
          </div>
          <div class="result-card result-card--wide" id="result-routine">
            <div class="result-icon">✨</div>
            <h4>Routine recommandée</h4>
            <p id="routine-text">—</p>
          </div>
          <div class="result-card result-card--wide" id="result-products">
            <div class="result-icon">🌿</div>
            <h4>Ingrédients clés à privilégier</h4>
            <p id="products-text">—</p>
          </div>
        </div>

        <div class="results-cta">
          <a href="{{ section.settings.shop_link }}" class="btn-shop">
            {{ section.settings.shop_btn_text }}
          </a>
          <button class="btn-restart" onclick="resetAnalyzer()">
            Analyser une autre photo
          </button>
        </div>
      </div>

      <!-- Step 4: Error -->
      <div class="step step-error hidden" id="step-error">
        <div class="error-content">
          <div class="error-icon">⚠️</div>
          <p id="error-message">Une erreur est survenue. Veuillez réessayer.</p>
          <button class="btn-restart" onclick="resetAnalyzer()">Réessayer</button>
        </div>
      </div>
    </div>

  </div>
</section>

<script>
  window.AVRIL_API_URL = "https://web-production-fb65c.up.railway.app";
</script>
<script src="{{ 'skin-analyzer.js' | asset_url }}" defer></script>

{% schema %}
{
  "name": "Skin Analyzer",
  "tag": "section",
  "class": "section",
  "settings": [
    {
      "type": "text",
      "id": "title",
      "label": "Titre",
      "default": "Analysez votre peau en 30 secondes"
    },
    {
      "type": "text",
      "id": "subtitle",
      "label": "Sous-titre",
      "default": "Notre IA analyse votre type de peau et vous recommande les soins adaptés parmi notre sélection."
    },
    {
      "type": "text",
      "id": "upload_text",
      "label": "Texte zone upload",
      "default": "Glissez votre photo ici ou cliquez pour choisir"
    },
    {
      "type": "text",
      "id": "upload_btn_text",
      "label": "Bouton upload",
      "default": "Choisir une photo"
    },
    {
      "type": "text",
      "id": "analyze_btn_text",
      "label": "Bouton analyser",
      "default": "Analyser ma peau"
    },
    {
      "type": "text",
      "id": "loading_text",
      "label": "Texte chargement",
      "default": "Analyse en cours... notre IA examine votre peau"
    },
    {
      "type": "text",
      "id": "shop_btn_text",
      "label": "Bouton boutique",
      "default": "Découvrir mes soins personnalisés"
    },
    {
      "type": "url",
      "id": "shop_link",
      "label": "Lien boutique",
      "default": "/collections/all"
    }
  ],
  "presets": [
    {
      "name": "Skin Analyzer"
    }
  ]
}
{% endschema %}
`,
  'assets/skin-analyzer.css': `/* ============================================
   Avril.ma — AI Skin Analyzer Styles
   ============================================ */

:root {
  --avril-pink: #e8a4b0;
  --avril-pink-dark: #d4768a;
  --avril-cream: #fdf8f5;
  --avril-text: #2d1b1e;
  --avril-text-light: #6b4a52;
  --avril-border: #f0dde3;
  --avril-white: #ffffff;
  --avril-shadow: 0 4px 24px rgba(180, 80, 100, 0.08);
  --avril-radius: 16px;
  --avril-radius-sm: 10px;
}

/* ---- Container ---- */
.avril-skin-analyzer {
  padding: 60px 20px;
  background: var(--avril-cream);
}

.analyzer-container {
  max-width: 760px;
  margin: 0 auto;
}

/* ---- Header ---- */
.analyzer-header {
  text-align: center;
  margin-bottom: 40px;
}

.analyzer-title {
  font-size: clamp(24px, 4vw, 36px);
  font-weight: 600;
  color: var(--avril-text);
  margin: 0 0 12px;
  line-height: 1.2;
}

.analyzer-subtitle {
  font-size: 16px;
  color: var(--avril-text-light);
  margin: 0;
  line-height: 1.6;
  max-width: 520px;
  margin: 0 auto;
}

/* ---- Upload Area ---- */
.upload-area {
  background: var(--avril-white);
  border: 2px dashed var(--avril-border);
  border-radius: var(--avril-radius);
  padding: 48px 32px;
  text-align: center;
  cursor: pointer;
  transition: border-color 0.2s, background 0.2s;
}

.upload-area:hover,
.upload-area.drag-over {
  border-color: var(--avril-pink);
  background: #fef5f7;
}

.upload-icon {
  color: var(--avril-pink);
  margin-bottom: 16px;
}

.upload-text {
  font-size: 16px;
  color: var(--avril-text);
  margin: 0 0 6px;
  font-weight: 500;
}

.upload-hint {
  font-size: 13px;
  color: var(--avril-text-light);
  margin: 0 0 24px;
}

/* ---- Buttons ---- */
.btn-upload {
  display: inline-block;
  background: var(--avril-pink);
  color: var(--avril-white);
  border: none;
  border-radius: 50px;
  padding: 14px 32px;
  font-size: 15px;
  font-weight: 500;
  cursor: pointer;
  transition: background 0.2s, transform 0.1s;
  font-family: inherit;
}

.btn-upload:hover {
  background: var(--avril-pink-dark);
  transform: translateY(-1px);
}

.btn-analyze {
  display: flex;
  align-items: center;
  gap: 10px;
  margin: 20px auto 0;
  background: var(--avril-text);
  color: var(--avril-white);
  border: none;
  border-radius: 50px;
  padding: 16px 40px;
  font-size: 16px;
  font-weight: 500;
  cursor: pointer;
  transition: background 0.2s, transform 0.1s;
  font-family: inherit;
}

.btn-analyze:hover {
  background: #1a0a0d;
  transform: translateY(-1px);
}

.btn-change-photo {
  display: flex;
  align-items: center;
  gap: 6px;
  background: transparent;
  color: var(--avril-text-light);
  border: 1px solid var(--avril-border);
  border-radius: 50px;
  padding: 8px 18px;
  font-size: 13px;
  cursor: pointer;
  margin: 12px auto 0;
  font-family: inherit;
  transition: border-color 0.2s;
}

.btn-change-photo:hover {
  border-color: var(--avril-pink);
  color: var(--avril-pink-dark);
}

.btn-shop {
  display: inline-block;
  background: var(--avril-pink);
  color: var(--avril-white);
  text-decoration: none;
  border-radius: 50px;
  padding: 16px 36px;
  font-size: 15px;
  font-weight: 500;
  transition: background 0.2s, transform 0.1s;
}

.btn-shop:hover {
  background: var(--avril-pink-dark);
  transform: translateY(-1px);
}

.btn-restart {
  background: transparent;
  color: var(--avril-text-light);
  border: 1px solid var(--avril-border);
  border-radius: 50px;
  padding: 14px 28px;
  font-size: 14px;
  cursor: pointer;
  font-family: inherit;
  transition: border-color 0.2s;
}

.btn-restart:hover {
  border-color: var(--avril-pink);
  color: var(--avril-pink-dark);
}

/* ---- Photo Preview ---- */
.photo-preview {
  text-align: center;
}

.photo-preview img {
  width: 180px;
  height: 180px;
  object-fit: cover;
  border-radius: 50%;
  border: 3px solid var(--avril-border);
  display: block;
  margin: 0 auto;
}

/* ---- Loading ---- */
.step-loading {
  text-align: center;
  padding: 60px 20px;
}

.loading-dots {
  display: flex;
  justify-content: center;
  gap: 8px;
  margin-bottom: 20px;
}

.loading-dots span {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background: var(--avril-pink);
  animation: loading-bounce 1.4s ease-in-out infinite both;
}

.loading-dots span:nth-child(1) { animation-delay: -0.32s; }
.loading-dots span:nth-child(2) { animation-delay: -0.16s; }

@keyframes loading-bounce {
  0%, 80%, 100% { transform: scale(0); opacity: 0.4; }
  40% { transform: scale(1); opacity: 1; }
}

.loading-text {
  color: var(--avril-text-light);
  font-size: 15px;
  margin: 0;
  font-style: italic;
}

/* ---- Results ---- */
.step-results {
  animation: fadeIn 0.5s ease;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

.results-header {
  text-align: center;
  margin-bottom: 28px;
}

.results-header h3 {
  font-size: 22px;
  color: var(--avril-text);
  margin: 0;
  font-weight: 600;
}

.results-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  margin-bottom: 32px;
}

.result-card {
  background: var(--avril-white);
  border-radius: var(--avril-radius-sm);
  padding: 24px;
  box-shadow: var(--avril-shadow);
  border: 1px solid var(--avril-border);
}

.result-card--wide {
  grid-column: 1 / -1;
}

.result-icon {
  font-size: 24px;
  margin-bottom: 10px;
}

.result-card h4 {
  font-size: 13px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--avril-text-light);
  margin: 0 0 8px;
  font-weight: 600;
}

.result-card p {
  font-size: 15px;
  color: var(--avril-text);
  margin: 0;
  line-height: 1.6;
}

.results-cta {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
}

/* ---- Error ---- */
.step-error {
  text-align: center;
  padding: 48px 20px;
}

.error-icon {
  font-size: 40px;
  margin-bottom: 16px;
}

.error-content p {
  color: var(--avril-text-light);
  font-size: 15px;
  margin: 0 0 20px;
}

/* ---- Utility ---- */
.hidden {
  display: none !important;
}

/* ---- Responsive ---- */
@media (max-width: 480px) {
  .avril-skin-analyzer {
    padding: 40px 16px;
  }

  .upload-area {
    padding: 32px 20px;
  }

  .results-grid {
    grid-template-columns: 1fr;
  }

  .result-card--wide {
    grid-column: auto;
  }
}
`,
  'assets/skin-analyzer.js': `/**
 * Avril.ma — AI Skin Analyzer
 * Frontend logic: upload, send to Railway API, display results
 */

(function () {
  'use strict';

  const API_URL = window.AVRIL_API_URL || 'https://web-production-fb65c.up.railway.app';

  let selectedFile = null;

  // ---- Init ----
  document.addEventListener('DOMContentLoaded', function () {
    const input = document.getElementById('skin-photo-input');
    const uploadArea = document.getElementById('upload-area');

    if (!input || !uploadArea) return;

    // File input change
    input.addEventListener('change', function (e) {
      const file = e.target.files[0];
      if (file) handleFileSelected(file);
    });

    // Drag & drop
    uploadArea.addEventListener('dragover', function (e) {
      e.preventDefault();
      uploadArea.classList.add('drag-over');
    });

    uploadArea.addEventListener('dragleave', function () {
      uploadArea.classList.remove('drag-over');
    });

    uploadArea.addEventListener('drop', function (e) {
      e.preventDefault();
      uploadArea.classList.remove('drag-over');
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith('image/')) {
        handleFileSelected(file);
      }
    });

    // Click on upload area (excluding button)
    uploadArea.addEventListener('click', function (e) {
      if (e.target === uploadArea || e.target.closest('.upload-icon') || e.target.closest('.upload-text') || e.target.closest('.upload-hint')) {
        input.click();
      }
    });
  });

  // ---- Handle file selection ----
  function handleFileSelected(file) {
    // Validate size (5 MB max)
    if (file.size > 5 * 1024 * 1024) {
      showError('Image trop grande. Veuillez choisir une image de moins de 5 MB.');
      return;
    }

    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = function (e) {
      const previewImg = document.getElementById('preview-img');
      const photoPreview = document.getElementById('photo-preview');
      const uploadArea = document.getElementById('upload-area');
      const btnAnalyze = document.getElementById('btn-analyze');

      if (previewImg) previewImg.src = e.target.result;
      if (photoPreview) photoPreview.classList.remove('hidden');
      if (uploadArea) uploadArea.classList.add('hidden');
      if (btnAnalyze) btnAnalyze.classList.remove('hidden');
    };
    reader.readAsDataURL(file);
  }

  // ---- Analyze photo ----
  window.analyzePhoto = function () {
    if (!selectedFile) return;

    showStep('loading');

    const formData = new FormData();
    formData.append('image', selectedFile);

    fetch(API_URL + '/api/analyze', {
      method: 'POST',
      body: formData
    })
      .then(function (response) {
        if (!response.ok) {
          return response.json().then(function (err) {
            throw new Error(err.error || 'Erreur serveur (' + response.status + ')');
          });
        }
        return response.json();
      })
      .then(function (data) {
        displayResults(data);
      })
      .catch(function (err) {
        console.error('Skin analyzer error:', err);
        showError(err.message || 'Une erreur est survenue. Veuillez réessayer.');
      });
  };

  // ---- Display results ----
  function displayResults(data) {
    // skin_type
    const skinTypeEl = document.getElementById('skin-type-text');
    if (skinTypeEl && data.skin_type) {
      skinTypeEl.textContent = data.skin_type;
    }

    // concerns
    const concernsEl = document.getElementById('concerns-text');
    if (concernsEl && data.concerns) {
      if (Array.isArray(data.concerns)) {
        concernsEl.textContent = data.concerns.join(', ');
      } else {
        concernsEl.textContent = data.concerns;
      }
    }

    // routine
    const routineEl = document.getElementById('routine-text');
    if (routineEl && data.routine) {
      if (Array.isArray(data.routine)) {
        routineEl.textContent = data.routine.join(' • ');
      } else {
        routineEl.textContent = data.routine;
      }
    }

    // recommended_ingredients or products
    const productsEl = document.getElementById('products-text');
    if (productsEl) {
      const ingredients = data.recommended_ingredients || data.key_ingredients || data.products;
      if (ingredients) {
        if (Array.isArray(ingredients)) {
          productsEl.textContent = ingredients.join(', ');
        } else {
          productsEl.textContent = ingredients;
        }
      }
    }

    showStep('results');
  }

  // ---- Show error ----
  function showError(message) {
    const errorEl = document.getElementById('error-message');
    if (errorEl) errorEl.textContent = message;
    showStep('error');
  }

  // ---- Step management ----
  function showStep(stepName) {
    const steps = ['upload', 'loading', 'results', 'error'];
    steps.forEach(function (s) {
      const el = document.getElementById('step-' + s);
      if (el) {
        if (s === stepName) {
          el.classList.remove('hidden');
        } else {
          el.classList.add('hidden');
        }
      }
    });
  }

  // ---- Reset ----
  window.resetAnalyzer = function () {
    selectedFile = null;

    const input = document.getElementById('skin-photo-input');
    const photoPreview = document.getElementById('photo-preview');
    const uploadArea = document.getElementById('upload-area');
    const btnAnalyze = document.getElementById('btn-analyze');

    if (input) input.value = '';
    if (photoPreview) photoPreview.classList.add('hidden');
    if (uploadArea) uploadArea.classList.remove('hidden');
    if (btnAnalyze) btnAnalyze.classList.add('hidden');

    showStep('upload');
  };

})();
`,
};

app.get('/setup', async (req, res) => {
  const { code } = req.query;
  if (!code) return res.status(400).json({ error: 'Param ?code= manquant' });
  try {
    const tokenResp = await shopifyPost('/admin/oauth/access_token', {
      client_id: CLIENT_ID, client_secret: CLIENT_SECRET, code
    });
    const tokenData = JSON.parse(tokenResp.body);
    if (!tokenData.access_token)
      return res.status(400).json({ error: 'Token exchange echoue', raw: tokenResp.body });

    const token = tokenData.access_token;
    const results = [];
    for (const [key, value] of Object.entries(THEME_FILES)) {
      const r = await shopifyPut(
        `/admin/api/2024-01/themes/${SHOPIFY_THEME_ID}/assets.json`,
        { asset: { key, value } }, token
      );
      const ok = r.status === 200;
      results.push({ key, status: r.status, ok });
      console.log(`[SETUP] ${ok ? 'OK' : 'FAIL'} ${key} (${r.status})`);
    }
    const allOk = results.every(r => r.ok);
    res.json({ success: allOk, token: token.slice(0,8)+'...', results });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// ── Add skin-analyzer section to homepage ───────────────
function shopifyGet(path, token) {
  return new Promise((resolve, reject) => {
    const req = https.request({
      hostname: SHOPIFY_SHOP, path, method: 'GET',
      headers: { 'X-Shopify-Access-Token': token }
    }, (res) => {
      let d = '';
      res.on('data', c => d += c);
      res.on('end', () => resolve({ status: res.statusCode, body: d }));
    });
    req.on('error', reject); req.end();
  });
}

app.get('/add-section', async (req, res) => {
  const { code } = req.query;
  if (!code) return res.status(400).json({ error: 'Param ?code= manquant' });
  try {
    const tokenResp = await shopifyPost('/admin/oauth/access_token', {
      client_id: CLIENT_ID, client_secret: CLIENT_SECRET, code
    });
    const tokenData = JSON.parse(tokenResp.body);
    if (!tokenData.access_token)
      return res.status(400).json({ error: 'Token exchange echoue', raw: tokenResp.body });

    const token = tokenData.access_token;
    const assetPath = '/admin/api/2024-01/themes/' + SHOPIFY_THEME_ID + '/assets.json?asset%5Bkey%5D=templates%2Findex.json';
    const getResp = await shopifyGet(assetPath, token);
    const getData = JSON.parse(getResp.body);
    if (!getData.asset) return res.status(400).json({ error: 'Template not found', raw: getResp.body });

    const template = JSON.parse(getData.asset.value);
    const sectionId = 'skin-analyzer-' + Date.now();
    template.sections[sectionId] = {
      type: 'skin-analyzer',
      settings: {
        title: 'Analysez votre peau avec l\'IA',
        subtitle: 'Obtenez une analyse personnalisee en 30 secondes'
      }
    };
    if (!template.order) template.order = [];
    template.order.push(sectionId);

    const putPath = '/admin/api/2024-01/themes/' + SHOPIFY_THEME_ID + '/assets.json';
    const putResp = await shopifyPut(
      putPath,
      { asset: { key: 'templates/index.json', value: JSON.stringify(template, null, 2) } },
      token
    );
    const ok = putResp.status === 200;
    res.json({ success: ok, sectionId, status: putResp.status });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// ── Health check ─────────────────────────────────────────
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'Avril.ma Skin Analyzer API',
    version: '1.0.0',
    openai_configured: !!process.env.OPENAI_API_KEY
  });
});

// ── Démarrage ─────────────────────────────────────────────
app.listen(port, () => {
  console.log(`
╔══════════════════════════════════════════════╗
║  🌸 Avril.ma Skin Analyzer — Serveur démarré ║
║  ➜  http://localhost:${port}                    ║
║  ➜  POST /api/analyze  (upload photo)        ║
║  ➜  GET  /api/health   (vérification)        ║
╚══════════════════════════════════════════════╝
  `);
  if (!process.env.OPENAI_API_KEY) {
    console.warn('⚠️  OPENAI_API_KEY non définie ! Ajoutez-la dans le fichier .env');
  }
});
