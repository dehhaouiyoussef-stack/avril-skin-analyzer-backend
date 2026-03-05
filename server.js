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
