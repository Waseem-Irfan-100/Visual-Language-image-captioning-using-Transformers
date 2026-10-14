import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ArrowStyle

def create_architecture_diagram(output_path_png='figures/fig1_architecture.png', output_path_pdf='figures/fig1_architecture.pdf'):
    fig = plt.figure(figsize=(15, 8.5), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Styling colors - Academic grayscale & elegant muted slate/navy/teal
    c_bg = '#FAFAFA'
    c_border = '#333333'
    c_vision_bg = '#EDF2F7'
    c_vision_box = '#E2E8F0'
    c_vision_border = '#4A5568'
    c_dec_bg = '#F7FAFC'
    c_dec_box = '#FFFFFF'
    c_dec_border = '#2D3748'
    c_sub_box = '#EDF2F7'
    c_accent_box = '#EBF8FF'
    c_accent_border = '#2B6CB0'
    c_out_box = '#F0FFF4'
    c_out_border = '#276749'
    c_arrow = '#2D3748'

    fig.patch.set_facecolor(c_bg)
    ax.set_facecolor(c_bg)

    def draw_box(x, y, w, h, text, title='', subtext='', bg=c_vision_box, border=c_vision_border, lw=1.5, boxstyle="round,pad=0.5", fontsize=9, title_fs=10, sub_fs=8):
        box = FancyBboxPatch((x, y), w, h, boxstyle=boxstyle, facecolor=bg, edgecolor=border, linewidth=lw, zorder=3)
        ax.add_patch(box)
        
        # Calculate vertical positioning
        cy = y + h / 2.0
        cx = x + w / 2.0
        
        if title and subtext:
            ax.text(cx, y + h * 0.72, title, ha='center', va='center', fontsize=title_fs, fontweight='bold', color='#1A202C', zorder=4)
            ax.text(cx, y + h * 0.45, text, ha='center', va='center', fontsize=fontsize, color='#2D3748', zorder=4)
            ax.text(cx, y + h * 0.20, subtext, ha='center', va='center', fontsize=sub_fs, fontstyle='italic', color='#4A5568', zorder=4)
        elif title:
            ax.text(cx, y + h * 0.65, title, ha='center', va='center', fontsize=title_fs, fontweight='bold', color='#1A202C', zorder=4)
            ax.text(cx, y + h * 0.32, text, ha='center', va='center', fontsize=fontsize, color='#2D3748', zorder=4)
        else:
            if subtext:
                ax.text(cx, y + h * 0.60, text, ha='center', va='center', fontsize=fontsize, fontweight='bold', color='#1A202C', zorder=4)
                ax.text(cx, y + h * 0.30, subtext, ha='center', va='center', fontsize=sub_fs, fontstyle='italic', color='#4A5568', zorder=4)
            else:
                ax.text(cx, cy, text, ha='center', va='center', fontsize=fontsize, fontweight='bold', color='#1A202C', zorder=4)
        return box

    def draw_arrow(x1, y1, x2, y2, label='', label_pos=(0.5, 0.5), color=c_arrow, lw=1.5, style='-|>', ls='-'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw, ls=ls, mutation_scale=14),
                    zorder=5)
        if label:
            lx = x1 + (x2 - x1) * label_pos[0]
            ly = y1 + (y2 - y1) * label_pos[1]
            ax.text(lx, ly, label, ha='center', va='center', fontsize=7.5, color='#4A5568',
                    bbox=dict(boxstyle='square,pad=0.2', facecolor=c_bg, edgecolor='none', alpha=0.9), zorder=6)

    # -------------------------------------------------------------
    # SECTION 1: VISION ENCODER (LEFT COLUMN)
    # -------------------------------------------------------------
    # Container boundary for Vision Branch
    v_box = FancyBboxPatch((4, 6), 28, 88, boxstyle="round,pad=1.0", facecolor='#F7FAFC', edgecolor='#CBD5E0', linewidth=1.2, linestyle='--', zorder=1)
    ax.add_patch(v_box)
    ax.text(18, 91.5, "VISUAL ENCODER BRANCH (Frozen)", ha='center', va='center', fontsize=10.5, fontweight='bold', color='#2B6CB0', zorder=2)

    # 1.1 Input Image
    draw_box(7, 76, 22, 10, "Input Image", title="", subtext="224 × 224 × 3 RGB", bg='#FFFFFF', border='#4A5568', lw=1.5)

    # Arrow down
    draw_arrow(18, 76, 18, 68, "16 × 16 Patching")

    # 1.2 Patch Extraction & Linear Projection
    draw_box(7, 57, 22, 11, "Patch Extraction & Emb.", title="Patch Partitioning", subtext="N = 196 Patches + [CLS]\nDim: 197 × 768", bg='#EDF2F7', border='#4A5568')

    # Arrow down
    draw_arrow(18, 57, 18, 49, "Patches + Position Enc.")

    # 1.3 Pretrained ViT-B/16
    draw_box(6, 29, 24, 20, "12 Transformer Encoder Layers\n• Multi-Head Self-Attention (12 heads)\n• MLP Blocks (dim = 3072)\n• Pre-LayerNorm & Residuals\n• 86M Parameters (Frozen)",
             title="Vision Transformer (ViT-B/16)", subtext="google/vit-base-patch16-224-in21k", bg='#EBF8FF', border='#2B6CB0', lw=2, fontsize=8, title_fs=9.5)

    # Arrow down
    draw_arrow(18, 29, 18, 21, "197 × 768 Visual Features")

    # 1.4 Projection Bridge
    draw_box(7, 11, 22, 10, "Projection Bridge", title="", subtext="Linear (768 → 768) + LayerNorm\nDecouples Encoder / Decoder", bg='#FFFFFF', border='#2B6CB0', lw=1.5)

    # -------------------------------------------------------------
    # SECTION 2: TRANSFORMER DECODER (CENTER-RIGHT REGION)
    # -------------------------------------------------------------
    # Container boundary for Decoder Branch
    d_box = FancyBboxPatch((35, 6), 40, 88, boxstyle="round,pad=1.0", facecolor='#F7FAFC', edgecolor='#CBD5E0', linewidth=1.2, linestyle='--', zorder=1)
    ax.add_patch(d_box)
    ax.text(55, 91.5, "TRANSFORMER DECODER (Trained from Scratch)", ha='center', va='center', fontsize=10.5, fontweight='bold', color='#2D3748', zorder=2)

    # 2.1 Input Tokens (Bottom of Decoder Column)
    draw_box(40, 11, 30, 9, "Caption Tokens (Teacher Forcing)", title="", subtext="[CLS], w₁, w₂, ..., wₜ₋₁ (Max len = 50)", bg='#FFFFFF', border='#4A5568')

    # Arrow Up
    draw_arrow(55, 20, 55, 26, "Text Tokens")

    # 2.2 WordPiece Embedding & Learned Positional Encoding
    draw_box(38, 26, 34, 10, "WordPiece Embedding + Pos. Enc.", title="BERT Tokenizer Embeddings", subtext="bert-base-uncased (Vocab: 30,522, d = 768)", bg='#EDF2F7', border='#4A5568')

    # Arrow Up
    draw_arrow(55, 36, 55, 42, "Word Vectors (d = 768)")

    # 2.3 Custom Decoder Block (x 6 Stacked Layers)
    dec_stack = FancyBboxPatch((37, 42), 36, 40, boxstyle="round,pad=0.8", facecolor='#FFFFFF', edgecolor='#2D3748', linewidth=2.0, zorder=2)
    ax.add_patch(dec_stack)
    ax.text(55, 80.2, "Custom Pre-LN Decoder Layer (× 6 Stack)", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#1A202C', zorder=3)
    ax.text(55, 78.2, "d_model = 768, heads = 8, d_ff = 2048, dropout = 0.1", ha='center', va='center', fontsize=7.5, fontstyle='italic', color='#4A5568', zorder=3)

    # Inside Decoder Layer:
    # Sub-block 1: Masked Self-Attention
    draw_box(39, 44, 32, 7.5, "Masked Multi-Head Self-Attention", title="", subtext="Causal Attention Mask (Triangular)", bg='#F7FAFC', border='#718096', lw=1.2, fontsize=8, sub_fs=7)

    # Arrow inside decoder
    draw_arrow(55, 51.5, 55, 55.5)

    # Sub-block 2: Multi-Head Cross Attention
    draw_box(39, 55.5, 32, 9, "Multi-Head Cross-Attention (h = 8)", title="", subtext="Queries: Text | Keys & Values: Visual Memory\nNative extraction of attention matrices α", bg='#EBF8FF', border='#2B6CB0', lw=1.5, fontsize=8, sub_fs=7)

    # Arrow inside decoder
    draw_arrow(55, 64.5, 55, 68.5)

    # Sub-block 3: Feed Forward
    draw_box(39, 68.5, 32, 7.5, "Feed-Forward Network (FFN)", title="", subtext="Linear (768 → 2048) → GELU → Linear (2048 → 768)", bg='#F7FAFC', border='#718096', lw=1.2, fontsize=8, sub_fs=7)

    # Cross-modal connection from Visual Encoder Projection Bridge to Cross-Attention
    # Coordinate route: (29, 16) -> (33, 16) -> (33, 60) -> (39, 60)
    ax.plot([29, 33, 33, 39], [16, 16, 60, 60], color='#2B6CB0', lw=2.0, ls='-', zorder=4)
    ax.annotate('', xy=(39, 60), xytext=(36, 60),
                arrowprops=dict(arrowstyle='-|>', color='#2B6CB0', lw=2.0, mutation_scale=14), zorder=5)
    ax.text(33, 37, "Visual Memory\n(Keys & Values)\n197 × 768", ha='center', va='center', fontsize=7.5, fontweight='bold', color='#2B6CB0',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFFFFF', edgecolor='#2B6CB0', lw=1), zorder=6)

    # Cross-Attention export arrow (Interpretability Heatmaps)
    draw_arrow(71, 60, 77, 60, color='#805AD5', lw=1.5)
    draw_box(77.5, 55, 19, 10, "Attention Weights α", title="Visual Grounding", subtext="14 × 14 Spatial Heatmaps\nWord-to-Patch Alignment", bg='#FAF5FF', border='#805AD5', lw=1.2, fontsize=8, title_fs=8.5, sub_fs=7)

    # Arrow from Decoder Stack Up to Head
    draw_arrow(55, 82, 55, 85)

    # -------------------------------------------------------------
    # SECTION 3: PREDICTION & DECODING (RIGHT / TOP)
    # -------------------------------------------------------------
    # Boundary box for Inference / Prediction Head
    p_box = FancyBboxPatch((76, 6), 21.5, 45, boxstyle="round,pad=0.8", facecolor='#F7FAFC', edgecolor='#CBD5E0', linewidth=1.2, linestyle='--', zorder=1)
    ax.add_patch(p_box)
    ax.text(86.7, 48.5, "PREDICTION & DECODING", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#276749', zorder=2)

    # Route from Decoder top (55, 85) to Weight-Tied Output Layer
    # Coordinates: (55, 85) -> (55, 88) -> (86.7, 88) -> (86.7, 43)
    ax.plot([55, 55, 86.7, 86.7], [82, 86, 86, 44], color=c_arrow, lw=1.5, zorder=4)
    ax.annotate('', xy=(86.7, 44), xytext=(86.7, 45),
                arrowprops=dict(arrowstyle='-|>', color=c_arrow, lw=1.5, mutation_scale=12), zorder=5)
    ax.text(71, 87.5, "Hidden States h_t (d = 768)", ha='center', va='center', fontsize=7.5, color='#4A5568', zorder=6)

    # 3.1 Weight Tied Linear Head
    draw_box(77.5, 34, 18.5, 9, "Linear Output Projection", title="Tied Linear Layer", subtext="Shared weights (768 → 30,522)\nVocabulary Logits", bg='#FFFFFF', border='#276749', lw=1.5, fontsize=8, title_fs=8.5, sub_fs=7)

    # Arrow down
    draw_arrow(86.7, 34, 86.7, 27)

    # 3.2 Decoding Strategy Box
    draw_box(77.5, 19, 18.5, 8, "Greedy (11.1 ms/img)\nBeam Search k=3 (146 ms/img)", title="Decoding Strategy", subtext="Length Penalty α = 0.7", bg='#F0FFF4', border='#276749', lw=1.5, fontsize=7.5, title_fs=8.5, sub_fs=7)

    # Arrow down
    draw_arrow(86.7, 19, 86.7, 13)

    # 3.3 Final Output Caption
    draw_box(76.5, 7.5, 20.5, 5.5, "\"A dog running across a green lawn.\"", title="", subtext="Emitted Natural Language Caption", bg='#FFFFFF', border='#276749', lw=1.8, fontsize=8, sub_fs=7)

    # Save outputs
    plt.savefig(output_path_png, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.savefig(output_path_pdf, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f'Figure 1 saved to: {output_path_png} and {output_path_pdf}')

if __name__ == '__main__':
    create_architecture_diagram()
