from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm, mm
from reportlab.lib.colors import HexColor, white, black
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
from reportlab.platypus.flowables import HRFlowable
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY, TA_RIGHT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import datetime

# Color scheme
PRIMARY_COLOR = HexColor('#0066CC')
SECONDARY_COLOR = HexColor('#009688')
ACCENT_COLOR = HexColor('#EE4C2C')
DARK_GRAY = HexColor('#404040')
LIGHT_GRAY = HexColor('#F0F0F0')
MEDIUM_GRAY = HexColor('#808080')

def create_pdf():
    # Create document
    doc = SimpleDocTemplate(
        "LMDIS_Project_Overview.pdf",
        pagesize=A4,
        rightMargin=2.5*cm,
        leftMargin=2.5*cm,
        topMargin=2.5*cm,
        bottomMargin=2.5*cm
    )
    
    # Story container
    story = []
    
    # Define styles
    styles = getSampleStyleSheet()
    
    # Title style
    styles.add(ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=32,
        textColor=PRIMARY_COLOR,
        spaceAfter=6,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    ))
    
    # Subtitle style
    styles.add(ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontSize=18,
        textColor=DARK_GRAY,
        spaceAfter=12,
        alignment=TA_CENTER,
        fontName='Helvetica'
    ))
    
    # Heading 1 style
    styles.add(ParagraphStyle(
        'Heading1Custom',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=PRIMARY_COLOR,
        spaceBefore=20,
        spaceAfter=10,
        fontName='Helvetica-Bold'
    ))
    
    # Heading 2 style
    styles.add(ParagraphStyle(
        'Heading2Custom',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=SECONDARY_COLOR,
        spaceBefore=15,
        spaceAfter=8,
        fontName='Helvetica-Bold'
    ))
    
    # Heading 3 style
    styles.add(ParagraphStyle(
        'Heading3Custom',
        parent=styles['Heading3'],
        fontSize=13,
        textColor=DARK_GRAY,
        spaceBefore=12,
        spaceAfter=6,
        fontName='Helvetica-Bold'
    ))
    
    # Body text style
    styles.add(ParagraphStyle(
        'BodyCustom',
        parent=styles['Normal'],
        fontSize=11,
        textColor=DARK_GRAY,
        spaceAfter=8,
        alignment=TA_JUSTIFY,
        leading=16
    ))
    
    # Bullet style
    styles.add(ParagraphStyle(
        'BulletCustom',
        parent=styles['Normal'],
        fontSize=11,
        textColor=DARK_GRAY,
        leftIndent=20,
        spaceAfter=4,
        leading=15
    ))
    
    # Code style
    styles.add(ParagraphStyle(
        'CodeCustom',
        parent=styles['Normal'],
        fontSize=9,
        textColor=DARK_GRAY,
        fontName='Courier',
        leftIndent=15,
        spaceAfter=8,
        leading=12,
        backColor=LIGHT_GRAY
    ))
    
    # Small text style
    styles.add(ParagraphStyle(
        'SmallText',
        parent=styles['Normal'],
        fontSize=9,
        textColor=MEDIUM_GRAY,
        spaceAfter=4
    ))
    
    # ===== TITLE PAGE =====
    story.append(Spacer(1, 3*inch))
    story.append(Paragraph('LMDIS', styles['CustomTitle']))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph('Lossless Multimodal Document Intelligence System', styles['Subtitle']))
    story.append(Spacer(1, 0.5*inch))
    
    # Horizontal line
    story.append(HRFlowable(width="100%", thickness=2, color=PRIMARY_COLOR, spaceAfter=20))
    
    story.append(Paragraph('Graph Guided Evidence Aware Reasoning', 
                          ParagraphStyle('tagline', parent=styles['Subtitle'], 
                                       fontSize=16, textColor=SECONDARY_COLOR)))
    story.append(Paragraph('for Enterprise Document Intelligence', 
                          ParagraphStyle('tagline2', parent=styles['Subtitle'], 
                                       fontSize=16, textColor=SECONDARY_COLOR)))
    story.append(Spacer(1, 1*inch))
    
    story.append(Paragraph('Project Overview and Technical Documentation',
                          ParagraphStyle('docType', parent=styles['Subtitle'], 
                                       fontSize=13, textColor=DARK_GRAY)))
    story.append(Spacer(1, 0.5*inch))
    
    # Authors
    story.append(Paragraph('<b>Sachin S</b>   and   <b>Ayush Raj</b>',
                          ParagraphStyle('authors', parent=styles['Normal'],
                                       fontSize=12, textColor=DARK_GRAY, alignment=TA_CENTER)))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph('Capstone Project 2026',
                          ParagraphStyle('capstone', parent=styles['Normal'],
                                       fontSize=11, textColor=MEDIUM_GRAY, alignment=TA_CENTER)))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph('sachin.shiva1612@gmail.com  |  ayushraj0901@gmail.com',
                          ParagraphStyle('emails', parent=styles['Normal'],
                                       fontSize=10, textColor=MEDIUM_GRAY, alignment=TA_CENTER)))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph('April 2026  |  Version 1.0',
                          ParagraphStyle('date', parent=styles['Normal'],
                                       fontSize=11, textColor=MEDIUM_GRAY, alignment=TA_CENTER)))
    
    story.append(PageBreak())
    
    # ===== COPYRIGHT PAGE =====
    story.append(Spacer(1, 2*inch))
    story.append(Paragraph('<b>Copyright © 2026 Sachin S and Ayush Raj</b>',
                          ParagraphStyle('copyright', parent=styles['Normal'],
                                       fontSize=12, textColor=DARK_GRAY, alignment=TA_CENTER)))
    story.append(Spacer(1, 0.5*inch))
    
    copyright_text = (
        'This document is part of the LMDIS (Lossless Multimodal Document Intelligence System) project, '
        'licensed under the GNU General Public License Version 3 (GPL 3.0).'
    )
    story.append(Paragraph(copyright_text,
                          ParagraphStyle('copyrightBody', parent=styles['BodyCustom'],
                                       fontSize=10, alignment=TA_CENTER)))
    story.append(Spacer(1, 0.3*inch))
    
    license_text = (
        'You are free to use, modify, and distribute this software under the terms of the GNU GPL 3.0 license. '
        'For complete license text, visit https://www.gnu.org/licenses/gpl-3.0.html'
    )
    story.append(Paragraph(license_text,
                          ParagraphStyle('licenseBody', parent=styles['BodyCustom'],
                                       fontSize=10, alignment=TA_CENTER)))
    story.append(Spacer(1, 0.5*inch))
    
    story.append(Paragraph('<b>Repository:</b> https://github.com/nihcastics/LMDIS-with-Graph-Guided-Evidence-Aware-Reasoning',
                          ParagraphStyle('repo', parent=styles['Normal'],
                                       fontSize=10, textColor=DARK_GRAY, alignment=TA_CENTER)))
    story.append(Spacer(1, 0.5*inch))
    
    story.append(Paragraph('All trademarks and product names referenced are the property of their respective owners.',
                          ParagraphStyle('trademark', parent=styles['Normal'],
                                       fontSize=9, textColor=MEDIUM_GRAY, alignment=TA_CENTER)))
    
    story.append(PageBreak())
    
    # ===== TABLE OF CONTENTS =====
    story.append(Paragraph('Table of Contents', styles['Heading1Custom']))
    story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY_COLOR, spaceAfter=15))
    
    toc_items = [
        '1. Executive Summary',
        '2. System Architecture',
        '   2.1 Document Processing Pipeline',
        '   2.2 Query and Retrieval System',
        '3. Module Inventory',
        '4. Technology Stack',
        '5. Key Capabilities',
        '6. Citation Engine',
        '7. API Reference',
        '8. Performance Characteristics',
        '9. Use Cases',
        '10. Getting Started',
        '11. Development Roadmap',
        '12. Limitations',
        '13. Contact and Repository Information',
        '14. License and Copyright'
    ]
    
    for item in toc_items:
        story.append(Paragraph(item, styles['BodyCustom']))
    
    story.append(PageBreak())
    
    # ===== EXECUTIVE SUMMARY =====
    story.append(Paragraph('1. Executive Summary', styles['Heading1Custom']))
    story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY_COLOR, spaceAfter=15))
    
    exec_summary = (
        'LMDIS is an enterprise-grade document intelligence platform that addresses the critical challenge of '
        'extracting, structuring, and retrieving information from complex multimodal documents without data loss. '
        'The system combines lossless parsing techniques, graph-based knowledge representation, semantic retrieval, '
        'and evidence-grounded response generation to deliver traceable and accurate document analytics.'
    )
    story.append(Paragraph(exec_summary, styles['BodyCustom']))
    
    exec_summary2 = (
        'The platform ingests PDF documents in any format (digital, scanned, or hybrid), performs comprehensive '
        'content extraction preserving layout hierarchy and visual elements, constructs semantically enriched '
        'knowledge graphs, and serves natural language queries through a dual-path reasoning architecture that '
        'ensures factual accuracy through citation-backed responses.'
    )
    story.append(Paragraph(exec_summary2, styles['BodyCustom']))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph('<b>Key Achievement:</b> A production-ready 21-module deterministic pipeline that '
                          'transforms complex multimodal documents into queryable knowledge graphs with '
                          'evidence-grounded, citation-accurate answers.',
                          ParagraphStyle('highlight', parent=styles['BodyCustom'],
                                       textColor=PRIMARY_COLOR, leftIndent=10)))
    
    story.append(PageBreak())
    
    # ===== SYSTEM ARCHITECTURE =====
    story.append(Paragraph('2. System Architecture', styles['Heading1Custom']))
    story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY_COLOR, spaceAfter=15))
    
    story.append(Paragraph('2.1 Document Processing Pipeline', styles['Heading2Custom']))
    
    story.append(Paragraph('The document processing pipeline executes through four distinct stages:',
                          styles['BodyCustom']))
    
    stages = [
        ('<b>Stage 1: Document Ingestion and Parsing (M1-M6)</b>', 
         'Documents are registered with cryptographic fingerprinting, classified by type (digital, scanned, or mixed), '
         'and processed through lossless content extraction that preserves text, tables, images, and layout structure.'),
        ('<b>Stage 2: Semantic Processing and Graph Construction (M7-M9b)</b>',
         'A knowledge graph is constructed using NetworkX with 15+ edge types, enriched with semantic metadata, '
         'and indexed with 1024-dimensional embeddings for efficient retrieval.'),
        ('<b>Stage 3: Knowledge Extraction and Evidence Graph (M10-M11, LLM-K)</b>',
         'Claims are extracted using rule-based methods, integrated into an evidence graph with inter-claim relationships, '
         'and supplemented with LLM-derived knowledge for comprehensive query resolution.'),
        ('<b>Stage 4: Indexing and Storage</b>',
         'All processed data is serialized and stored in per-document isolated storage, enabling individual document '
         'management without cross-contamination.')
    ]
    
    for title, desc in stages:
        story.append(Paragraph(title, styles['Heading3Custom']))
        story.append(Paragraph(desc, styles['BodyCustom']))
    
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph('2.2 Query and Retrieval System', styles['Heading2Custom']))
    
    story.append(Paragraph('The query pipeline implements a sophisticated multi-stage retrieval and dual-path answer generation:',
                          styles['BodyCustom']))
    
    query_steps = [
        '<b>Query Interpretation (M12):</b> Intent detection and sub-claim decomposition',
        '<b>Multi-Strategy Retrieval (M13-15):</b> Bi-encoder search, graph traversal, cross-encoder re-ranking',
        '<b>Dual-Path Generation:</b> Evidence-bound answers (M16-19) and LLM Knowledge Store (Path B)',
        '<b>Answer Selection Layer (ASL):</b> Multi-signal scoring and arbitration',
        '<b>Formatting (M20):</b> Citation linking and confidence scoring',
        '<b>Suggestions (M21):</b> Context-aware follow-up questions'
    ]
    
    for i, step in enumerate(query_steps, 1):
        story.append(Paragraph(f'{i}. {step}', styles['BulletCustom']))
    
    story.append(PageBreak())
    
    # ===== MODULE INVENTORY =====
    story.append(Paragraph('3. Module Inventory', styles['Heading1Custom']))
    story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY_COLOR, spaceAfter=15))
    
    story.append(Paragraph('The system comprises 21 specialized modules organized into logical processing stages:',
                          styles['BodyCustom']))
    story.append(Spacer(1, 0.2*inch))
    
    # Module table
    module_data = [
        ['Module', 'Name', 'Function'],
        ['M1', 'Ingestion', 'Immutable source registration with SHA-256 fingerprinting'],
        ['M2', 'Page Type Classifier', 'Classifies pages as digital, scanned, or mixed'],
        ['M3', 'Content Extraction', 'Lossless text, table, and image extraction'],
        ['M4', 'Line Memory', 'Line-level normalization and paragraph detection'],
        ['M5', 'Structure Detection', 'Hierarchical section and heading recognition'],
        ['M6', 'Normalization', 'Component-level normalization'],
        ['M7', 'Graph Construction', 'NetworkX DiGraph with 15+ edge types'],
        ['M8', 'Confidence Annotation', 'Per-node OCR confidence scoring'],
        ['M9', 'Embedding Generation', '1024-dimensional embeddings with FAISS'],
        ['M9b', 'Semantic Enrichment', 'Information type classification (14 types)'],
        ['M10', 'Claim Extraction', 'Rule-based extraction of key claims'],
        ['M11', 'Evidence Graph', 'Claim integration and relationship mapping'],
        ['LLM-K', 'Knowledge Store', 'Direct LLM document comprehension'],
        ['M12', 'Query Interpretation', 'Intent detection and decomposition'],
        ['M13-15', 'Retrieval Pipeline', 'Bi-encoder, graph traversal, cross-encoder'],
        ['M16-19', 'Answer Generation', 'Evidence-bound LLM answer generation'],
        ['ASL', 'Answer Selection', 'Dual-path fact extraction and scoring'],
        ['M20', 'Formatter', 'Citation linking and confidence scoring'],
        ['M21', 'Suggestions', 'Context-aware follow-up generation']
    ]
    
    # Create table
    col_widths = [1.2*inch, 1.8*inch, 4*inch]
    module_table = Table(module_data, colWidths=col_widths)
    
    table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY_COLOR),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('BACKGROUND', (0, 1), (-1, -1), white),
        ('TEXTCOLOR', (0, 1), (-1, -1), DARK_GRAY),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, LIGHT_GRAY),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, LIGHT_GRAY]),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ])
    module_table.setStyle(table_style)
    story.append(module_table)
    
    story.append(PageBreak())
    
    # ===== TECHNOLOGY STACK =====
    story.append(Paragraph('4. Technology Stack', styles['Heading1Custom']))
    story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY_COLOR, spaceAfter=15))
    
    story.append(Paragraph('LMDIS leverages state-of-the-art technologies across the entire processing pipeline:',
                          styles['BodyCustom']))
    story.append(Spacer(1, 0.2*inch))
    
    tech_data = [
        ['Layer', 'Technology', 'Purpose'],
        ['Backend Framework', 'Python 3.13, FastAPI, Uvicorn', 'High-performance async API server'],
        ['Embedding Model', 'BAAI bge-large-en-v1.5', '1024-dimensional semantic embeddings'],
        ['Cross Encoder', 'ms-marco-MiniLM-L-12-v2', 'Evidence re-ranking for precision retrieval'],
        ['Vector Store', 'FAISS IndexFlatIP', 'Efficient similarity search'],
        ['Graph Engine', 'NetworkX DiGraph', 'Knowledge graph construction and traversal'],
        ['LLM Integration', 'OpenRouter, Google Gemini, Anthropic Claude', 'Multi-model LLM orchestration'],
        ['OCR Engine', 'PaddleOCR 3.4 with PyMuPDF', 'Scanned document text extraction'],
        ['PDF Parsing', 'pdfplumber, PyMuPDF', 'Lossless content extraction'],
        ['Frontend', 'HTML/CSS/JavaScript', 'Modern responsive interface'],
        ['Storage', 'JSON and Pickle serialization', 'Isolated document storage']
    ]
    
    tech_table = Table(tech_data, colWidths=col_widths)
    tech_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), SECONDARY_COLOR),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('BACKGROUND', (0, 1), (-1, -1), white),
        ('TEXTCOLOR', (0, 1), (-1, -1), DARK_GRAY),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, LIGHT_GRAY),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, LIGHT_GRAY]),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(tech_table)
    
    story.append(PageBreak())
    
    # ===== KEY CAPABILITIES =====
    story.append(Paragraph('5. Key Capabilities', styles['Heading1Custom']))
    story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY_COLOR, spaceAfter=15))
    
    capabilities = [
        ('<b>Lossless Structural Parsing</b>',
         'Preserves document layout, hierarchy, text formatting, table structures, and visual elements exactly as '
         'they appear in the source document. Adaptive extraction handles digital PDFs through PyMuPDF and '
         'pdfplumber, while scanned documents are processed through PaddleOCR with layout preservation.'),
        ('<b>Knowledge Graph Construction</b>',
         'Documents are transformed into directed graphs using NetworkX with 15+ edge types capturing semantic '
         'relationships, structural hierarchies, and cross-references. This graph representation enables sophisticated '
         'reasoning, traceability, and evidence-based query resolution.'),
        ('<b>Semantic Retrieval Engine</b>',
         'Replaces brittle keyword matching with embedding-based similarity search using BAAI bge-large-en-v1.5 '
         '(1024-dimensional vectors) indexed through FAISS. The retrieval pipeline combines bi-encoder search, '
         'graph traversal, and cross-encoder re-ranking for high-precision evidence extraction.'),
        ('<b>Evidence-Grounded Responses</b>',
         'Every answer is backed by verifiable source evidence with clear citations. The dual-path architecture '
         'generates answers from both the document knowledge graph and direct LLM reasoning, then arbitrates '
         'between them through a multi-signal Answer Selection Layer to ensure accuracy and prevent hallucinations.'),
        ('<b>Composite Confidence Scoring</b>',
         'Each citation carries a multi-factor confidence score blending semantic alignment (40%), OCR quality (30%), '
         'cross-encoder relevance (20%), and contribution verification (10%).')
    ]
    
    for title, desc in capabilities:
        story.append(Paragraph(title, styles['Heading2Custom']))
        story.append(Paragraph(desc, styles['BodyCustom']))
        story.append(Spacer(1, 0.15*inch))
    
    story.append(PageBreak())
    
    # ===== CITATION ENGINE =====
    story.append(Paragraph('6. Citation Engine', styles['Heading1Custom']))
    story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY_COLOR, spaceAfter=15))
    
    story.append(Paragraph('The citation engine provides transparent and verifiable source attribution for every claim '
                          'in the generated answer. Citations are constructed from the cross-encoder re-ranked evidence '
                          'pipeline rather than simple keyword matching, ensuring high-precision source linking.',
                          styles['BodyCustom']))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph('6.1 Multi-Signal Citation Attribution', styles['Heading2Custom']))
    
    attribution_signals = [
        '<b>Unigram Overlap (25%):</b> Content word matching between generated answer and evidence text',
        '<b>Bigram Overlap (30%):</b> Phrase-level attribution capturing multi-word concepts',
        '<b>High-Value Token Overlap (25%):</b> Matching of numbers, proper nouns, and abbreviations',
        '<b>Retrieval Alignment Prior (20%):</b> Score inherited from evidence retrieval pipeline'
    ]
    
    for signal in attribution_signals:
        story.append(Paragraph(signal, styles['BulletCustom']))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph('6.2 Citation Metadata', styles['Heading2Custom']))
    
    story.append(Paragraph('Every citation includes:', styles['BodyCustom']))
    
    metadata_items = [
        'Source text excerpt with page number reference',
        'Composite confidence score (0.0 to 1.0)',
        'Information type classification (fact, definition, statistic, methodology, result)',
        'Alignment score from the retrieval pipeline',
        'Contribution flag indicating independent attribution'
    ]
    
    for item in metadata_items:
        story.append(Paragraph(f'• {item}', styles['BulletCustom']))
    
    story.append(PageBreak())
    
    # ===== API REFERENCE =====
    story.append(Paragraph('7. API Reference', styles['Heading1Custom']))
    story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY_COLOR, spaceAfter=15))
    
    story.append(Paragraph('LMDIS exposes a RESTful API for programmatic access to all system capabilities:',
                          styles['BodyCustom']))
    story.append(Spacer(1, 0.2*inch))
    
    api_data = [
        ['Method', 'Endpoint', 'Description'],
        ['GET', '/', 'System health check and status'],
        ['GET', '/documents', 'List all uploaded documents'],
        ['POST', '/documents/upload', 'Upload and process PDF'],
        ['GET', '/documents/{id}/status', 'Check processing status'],
        ['POST', '/documents/{id}/query', 'Query with natural language'],
        ['GET', '/documents/{id}/suggestions', 'Get follow-up questions'],
        ['GET', '/documents/{id}/structure', 'Retrieve document structure']
    ]
    
    api_table = Table(api_data, colWidths=[1*inch, 2.5*inch, 3.5*inch])
    api_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), ACCENT_COLOR),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('BACKGROUND', (0, 1), (-1, -1), white),
        ('TEXTCOLOR', (0, 1), (-1, -1), DARK_GRAY),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, LIGHT_GRAY),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, LIGHT_GRAY]),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0, 1), (0, -1), PRIMARY_COLOR),
    ]))
    story.append(api_table)
    
    story.append(PageBreak())
    
    # ===== PERFORMANCE CHARACTERISTICS =====
    story.append(Paragraph('8. Performance Characteristics', styles['Heading1Custom']))
    story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY_COLOR, spaceAfter=15))
    
    story.append(Paragraph('8.1 Processing Time', styles['Heading2Custom']))
    
    processing_times = [
        '<b>Simple digital PDFs (10-50 pages):</b> 30 to 90 seconds',
        '<b>Complex digital PDFs (50-200 pages):</b> 2 to 5 minutes',
        '<b>Scanned documents (10-50 pages):</b> 2 to 4 minutes',
        '<b>Large scanned documents (50-200 pages):</b> 5 to 15 minutes'
    ]
    
    for time_item in processing_times:
        story.append(Paragraph(time_item, styles['BulletCustom']))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph('8.2 Query Response Time', styles['Heading2Custom']))
    
    query_times = [
        '<b>Simple factual queries:</b> 3 to 8 seconds',
        '<b>Complex analytical queries:</b> 8 to 15 seconds',
        '<b>Multi-aspect queries with graph traversal:</b> 10 to 20 seconds'
    ]
    
    for time_item in query_times:
        story.append(Paragraph(time_item, styles['BulletCustom']))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph('8.3 Resource Utilization', styles['Heading2Custom']))
    
    resources = [
        '<b>Memory:</b> 1.3 GB for embedding model + 50-200 MB per 100 pages',
        '<b>Storage:</b> 10-50 MB per 100 pages depending on content',
        '<b>CPU:</b> Multi-core systems show significant performance improvements',
        '<b>GPU:</b> Optional CUDA acceleration for embeddings and re-ranking'
    ]
    
    for resource in resources:
        story.append(Paragraph(resource, styles['BulletCustom']))
    
    story.append(PageBreak())
    
    # ===== USE CASES =====
    story.append(Paragraph('9. Use Cases', styles['Heading1Custom']))
    story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY_COLOR, spaceAfter=15))
    
    use_cases = [
        ('<b>Legal and Compliance Document Analysis</b>',
         'Analyze contracts, regulatory filings, and compliance documents with precise citation tracking. Extract '
         'obligations, deadlines, monetary terms, and compliance requirements with verifiable source attribution.'),
        ('<b>Financial Auditing and Reporting</b>',
         'Process financial statements, audit reports, and annual filings to extract financial metrics, performance '
         'indicators, and comparative analysis. Generate evidence-backed answers for audit queries with complete '
         'traceability.'),
        ('<b>Healthcare Record Processing</b>',
         'Analyze medical records, clinical trial reports, and research publications while maintaining data integrity. '
         'Extract patient information, treatment protocols, outcomes, and statistical findings with accurate citations.'),
        ('<b>Research and Academic Document Analysis</b>',
         'Process research papers, dissertations, and academic publications for literature review and knowledge '
         'synthesis. Extract methodologies, results, conclusions, and cross-references with academic rigor.'),
        ('<b>Enterprise Knowledge Management</b>',
         'Build searchable knowledge bases from corporate documentation, standard operating procedures, technical '
         'manuals, and policy documents. Enable employees to query organizational knowledge with source-verified answers.')
    ]
    
    for title, desc in use_cases:
        story.append(Paragraph(title, styles['Heading2Custom']))
        story.append(Paragraph(desc, styles['BodyCustom']))
        story.append(Spacer(1, 0.15*inch))
    
    story.append(PageBreak())
    
    # ===== GETTING STARTED =====
    story.append(Paragraph('10. Getting Started', styles['Heading1Custom']))
    story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY_COLOR, spaceAfter=15))
    
    story.append(Paragraph('10.1 Prerequisites', styles['Heading2Custom']))
    
    prereqs = [
        'Python 3.13 or higher',
        'Windows 10/11, Linux, or macOS',
        'Minimum 4 GB RAM',
        '2 GB available disk space',
        'Internet connection for API calls and model downloads',
        'At least one LLM API key (OpenRouter, Google Gemini, or Anthropic Claude)'
    ]
    
    for prereq in prereqs:
        story.append(Paragraph(f'• {prereq}', styles['BulletCustom']))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph('10.2 Installation', styles['Heading2Custom']))
    
    story.append(Paragraph('# Clone the repository', styles['CodeCustom']))
    story.append(Paragraph('git clone https://github.com/nihcastics/LMDIS-with-Graph-Guided-Evidence-Aware-Reasoning.git',
                          styles['CodeCustom']))
    story.append(Paragraph('cd LMDIS-with-Graph-Guided-Evidence-Aware-Reasoning', styles['CodeCustom']))
    story.append(Spacer(1, 0.1*inch))
    
    story.append(Paragraph('# Create virtual environment', styles['CodeCustom']))
    story.append(Paragraph('python -m venv .venv', styles['CodeCustom']))
    story.append(Paragraph('.venv\\Scripts\\activate  # Windows', styles['CodeCustom']))
    story.append(Spacer(1, 0.1*inch))
    
    story.append(Paragraph('# Install dependencies', styles['CodeCustom']))
    story.append(Paragraph('pip install -r backend/requirements.txt', styles['CodeCustom']))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph('10.3 Configuration', styles['Heading2Custom']))
    
    story.append(Paragraph('Set environment variables for API keys:', styles['BodyCustom']))
    story.append(Spacer(1, 0.1*inch))
    
    story.append(Paragraph('# Windows (PowerShell)', styles['CodeCustom']))
    story.append(Paragraph('$env:OPENROUTER_API_KEY="your_api_key_here"', styles['CodeCustom']))
    story.append(Paragraph('$env:GOOGLE_API_KEY="your_api_key_here"', styles['CodeCustom']))
    story.append(Paragraph('$env:ANTHROPIC_API_KEY="your_api_key_here"', styles['CodeCustom']))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph('10.4 Running the System', styles['Heading2Custom']))
    
    story.append(Paragraph('# Option 1: One-click launch (Windows) - Double-click start.bat', styles['CodeCustom']))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph('# Option 2: Manual launch', styles['CodeCustom']))
    story.append(Paragraph('uvicorn backend.app.main:app --host 127.0.0.1 --port 8000', styles['CodeCustom']))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph('Access the frontend at: http://127.0.0.1:8000/frontend/index.html',
                          ParagraphStyle('url', parent=styles['BodyCustom'], textColor=PRIMARY_COLOR)))
    
    story.append(PageBreak())
    
    # ===== DEVELOPMENT ROADMAP =====
    story.append(Paragraph('11. Development Roadmap', styles['Heading1Custom']))
    story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY_COLOR, spaceAfter=15))
    
    story.append(Paragraph('Planned Features:', styles['BodyCustom']))
    
    roadmap_items = [
        '<b>Multi-Document Reasoning:</b> Cross-document linking and comparative analysis',
        '<b>Performance Optimization:</b> Incremental processing and reduced memory footprint',
        '<b>Enhanced OCR:</b> Handwritten text recognition and complex layout detection',
        '<b>Real-Time Processing:</b> Streaming document processing and progressive query answering',
        '<b>Enterprise Features:</b> Role-based access control, audit logging, API rate limiting',
        '<b>Model Flexibility:</b> Pluggable embedding models and custom fine-tuning support'
    ]
    
    for item in roadmap_items:
        story.append(Paragraph(item, styles['BulletCustom']))
    
    story.append(PageBreak())
    
    # ===== LIMITATIONS =====
    story.append(Paragraph('12. Limitations', styles['Heading1Custom']))
    story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY_COLOR, spaceAfter=15))
    
    story.append(Paragraph('The current implementation has the following known limitations:', styles['BodyCustom']))
    
    limitations = [
        '<b>Computational Requirements:</b> Large documents require significant processing time and memory',
        '<b>OCR Quality Dependency:</b> Accuracy depends on input document quality for scanned content',
        '<b>Single Document Scope:</b> Multi-document reasoning planned for future releases',
        '<b>Model Download Requirements:</b> 1.5 GB model weights require internet connectivity for first run'
    ]
    
    for limit in limitations:
        story.append(Paragraph(limit, styles['BulletCustom']))
    
    story.append(PageBreak())
    
    # ===== CONTACT AND REPOSITORY =====
    story.append(Paragraph('13. Contact and Repository Information', styles['Heading1Custom']))
    story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY_COLOR, spaceAfter=15))
    
    story.append(Paragraph('13.1 Primary Contacts', styles['Heading2Custom']))
    
    story.append(Paragraph('<b>Sachin S:</b> sachin.shiva1612@gmail.com', styles['BodyCustom']))
    story.append(Paragraph('<b>Ayush Raj:</b> ayushraj0901@gmail.com', styles['BodyCustom']))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph('13.2 GitHub Repository', styles['Heading2Custom']))
    
    story.append(Paragraph('<b>Repository:</b> https://github.com/nihcastics/LMDIS-with-Graph-Guided-Evidence-Aware-Reasoning',
                          styles['BodyCustom']))
    story.append(Paragraph('<b>Issues:</b> https://github.com/nihcastics/LMDIS-with-Graph-Guided-Evidence-Aware-Reasoning/issues',
                          styles['BodyCustom']))
    story.append(Paragraph('<b>Discussions:</b> https://github.com/nihcastics/LMDIS-with-Graph-Guided-Evidence-Aware-Reasoning/discussions',
                          styles['BodyCustom']))
    
    story.append(PageBreak())
    
    # ===== LICENSE AND COPYRIGHT =====
    story.append(Paragraph('14. License and Copyright', styles['Heading1Custom']))
    story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY_COLOR, spaceAfter=15))
    
    story.append(Paragraph('14.1 GNU General Public License Version 3', styles['Heading2Custom']))
    
    story.append(Paragraph('This project is licensed under the GNU General Public License Version 3 (GPL 3.0). '
                          'You are free to use, modify, and distribute this software under the terms of the license.',
                          styles['BodyCustom']))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph('14.2 Copyright Notice', styles['Heading2Custom']))
    
    story.append(Paragraph('<b>Copyright © 2026 Sachin S and Ayush Raj</b>',
                          ParagraphStyle('copyrightNotice', parent=styles['BodyCustom'],
                                       textColor=PRIMARY_COLOR)))
    story.append(Spacer(1, 0.15*inch))
    
    story.append(Paragraph('This program is free software: you can redistribute it and/or modify it under the terms '
                          'of the GNU General Public License as published by the Free Software Foundation, either '
                          'version 3 of the License, or (at your option) any later version.',
                          styles['BodyCustom']))
    
    story.append(Paragraph('This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; '
                          'without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. '
                          'See the GNU General Public License for more details.',
                          styles['BodyCustom']))
    
    story.append(Paragraph('You should have received a copy of the GNU General Public License along with this program. '
                          'If not, see https://www.gnu.org/licenses/',
                          styles['BodyCustom']))
    
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph('14.3 Academic Use', styles['Heading2Custom']))
    
    story.append(Paragraph('LMDIS was developed as a capstone project demonstrating advanced document intelligence '
                          'capabilities. Academic use, research applications, and educational deployments are encouraged.',
                          styles['BodyCustom']))
    
    story.append(Spacer(1, 0.5*inch))
    
    # End of document
    story.append(HRFlowable(width="50%", thickness=1, color=DARK_GRAY, spaceAfter=15))
    story.append(Paragraph('<b>End of Document</b>',
                          ParagraphStyle('endDoc', parent=styles['Normal'],
                                       fontSize=14, textColor=PRIMARY_COLOR, alignment=TA_CENTER)))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph('LMDIS: Graph Guided Evidence Aware Reasoning for Enterprise Document Intelligence',
                          ParagraphStyle('taglineEnd', parent=styles['Normal'],
                                       fontSize=10, textColor=MEDIUM_GRAY, alignment=TA_CENTER)))
    story.append(Paragraph('Built with precision. Grounded in evidence. Designed for trust.',
                          ParagraphStyle('taglineEnd2', parent=styles['Normal'],
                                       fontSize=10, textColor=MEDIUM_GRAY, alignment=TA_CENTER)))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph('Copyright © 2026 Sachin S and Ayush Raj. Licensed under GPL 3.0.',
                          ParagraphStyle('copyrightEnd', parent=styles['Normal'],
                                       fontSize=9, textColor=MEDIUM_GRAY, alignment=TA_CENTER)))
    
    # Build PDF
    doc.build(story)
    print("PDF created successfully: LMDIS_Project_Overview.pdf")

if __name__ == '__main__':
    create_pdf()
