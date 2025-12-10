import React, { useState, useEffect } from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import '../css/homepage.css';

const HomePage = () => {
  const {siteConfig} = useDocusaurusContext();
  const isLightMode = false; // Always dark mode
  const [isScrolled, setIsScrolled] = useState(false);


  // Handle scroll for navbar effect
  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 50);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <Layout
      title={`${siteConfig.title}`}
      description="Physical AI & Humanoid Robotics Textbook - University-level educational content for modern robotics"
      wrapperClassName="homepage-layout">
      <div style={{
        minHeight: '100vh',
        fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
        background: isLightMode
          ? 'linear-gradient(135deg, #f9fafb 0%, #ffffff 50%, #f9fafb 100%)'
          : 'linear-gradient(135deg, #111827 0%, #000000 50%, #111827 100%)',
        color: isLightMode ? '#1f2937' : '#ffffff',
        transition: 'all 0.5s ease'
      }}>

        {/* Main Content */}
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: '85vh',
          padding: '3rem 0',
          maxWidth: '1400px',
          margin: '0 auto',
          width: '100%'
        }}>
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            gap: '5rem',
            width: '100%'
          }}>

            {/* Image Section (Left on Desktop) */}
            <div style={{
              width: '100%',
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
              marginBottom: '3rem'
            }}>
              <div style={{
                position: 'relative',
                width: '100%',
                maxWidth: '500px',
                height: '450px'
              }}>
                {/* Media query for tablet */}
                <style>{`
                  @media (min-width: 768px) {
                    div[style*="height: 450px"] {
                      height: 500px !important;
                    }
                  }

                  @media (min-width: 992px) {
                    div[style*="flex-direction: column"] {
                      flex-direction: row !important;
                    }

                    div[style*="margin-bottom: 3rem"] {
                      width: 55%;
                      margin-bottom: 0;
                    }

                    div[style*="text-align: center"] {
                      width: 45%;
                      text-align: left;
                    }
                  }

                  @media (max-width: 767px) {
                    h1[style*="font-size: 3rem"] {
                      font-size: 2.5rem !important;
                    }
                  }
                `}</style>

                {/* Gradient Border */}
                <div style={{
                  position: 'absolute',
                  top: '-2px',
                  left: '-2px',
                  right: '-2px',
                  bottom: '-2px',
                  background: 'linear-gradient(45deg, #E00070, #FF7A00)',
                  borderRadius: '15px',
                  zIndex: 0,
                  opacity: 0.8
                }}></div>

                {/* Book Image */}
                <div style={{
                  width: '100%',
                  height: '100%',
                  backgroundImage: "url('/img/hero-img.png')",
                  backgroundSize: 'cover',
                  backgroundPosition: 'center',
                  borderRadius: '13px',
                  position: 'relative',
                  zIndex: 1,
                  overflow: 'hidden',
                  border: `1px solid ${isLightMode ? 'rgba(224, 0, 112, 0.2)' : 'rgba(224, 0, 112, 0.3)'}`,
                  transition: 'all 0.3s ease',
                  filter: isLightMode ? 'brightness(1) contrast(1.1)' : 'brightness(1) contrast(1)'
                }}
                onMouseEnter={(e) => {
                  e.target.style.transform = 'scale(1.02)';
                  e.target.style.boxShadow = '0 15px 40px rgba(0, 0, 0, 0.4)';
                }}
                onMouseLeave={(e) => {
                  e.target.style.transform = 'scale(1)';
                  e.target.style.boxShadow = 'none';
                }}
                >
                  {/* Image overlay for better visibility */}
                  <div style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    right: 0,
                    bottom: 0,
                    background: isLightMode
                      ? 'linear-gradient(135deg, rgba(0, 0, 0, 0.15) 0%, rgba(0, 0, 0, 0.25) 100%)'
                      : 'linear-gradient(135deg, rgba(0, 0, 0, 0.2) 0%, rgba(0, 0, 0, 0) 100%)',
                    borderRadius: '13px',
                    pointerEvents: 'none'
                  }}></div>
                </div>
              </div>
            </div>

            {/* Text Section (Right on Desktop) */}
            <div style={{
              width: '100%',
              textAlign: 'left'
            }}>
              {/* Main Title - Exactly 2 lines */}
              <h1 style={{
                fontSize: '3rem',
                fontWeight: '800',
                background: 'linear-gradient(45deg, #E00070, #FF7A00)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                backgroundClip: 'text',
                marginBottom: '1.8rem',
                lineHeight: '1.1',
                letterSpacing: '-0.5px',
                maxWidth: '800px'
              }}>
                Physical AI &<br />Humanoid Robotics Textbook
              </h1>

              {/* AI Native Software Development */}
              <div style={{
                fontSize: '2rem',
                fontWeight: '700',
                color: isLightMode ? '#1f2937' : 'white',
                marginBottom: '1.5rem',
                lineHeight: '1.2'
              }}>
                AI Native Software Development
              </div>

              {/* Colearning Agentic AI */}
              <div style={{
                fontSize: '1.6rem',
                color: isLightMode ? '#4b5563' : '#d1d5db',
                marginBottom: '2.5rem',
                lineHeight: '1.4',
                fontWeight: '600'
              }}>
                Colearning Agentic AI<br />with Python and TypeScript
              </div>

              {/* Buttons Container */}
              <div style={{
                display: 'flex',
                gap: '1.2rem',
                justifyContent: 'center',
                flexWrap: 'wrap',
                marginTop: '2.5rem'
              }}>
                <a
                  href="https://github.com/MoattarAnsari385"
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{
                    padding: '1rem 2.5rem',
                    borderRadius: '10px',
                    background: 'linear-gradient(45deg, #E00070, #FF7A00)',
                    color: 'white',
                    fontWeight: '600',
                    textDecoration: 'none',
                    transition: 'all 0.3s ease',
                    display: 'inline-flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '0.7rem',
                    minWidth: '180px',
                    boxShadow: '0 4px 15px rgba(224, 0, 112, 0.3)'
                  }}
                  onMouseEnter={(e) => {
                    e.target.style.transform = 'translateY(-3px)';
                    e.target.style.boxShadow = '0 8px 25px rgba(224, 0, 112, 0.4)';
                  }}
                  onMouseLeave={(e) => {
                    e.target.style.transform = 'translateY(0)';
                    e.target.style.boxShadow = '0 4px 15px rgba(224, 0, 112, 0.3)';
                  }}
                  onClick={(e) => {
                    e.target.style.transform = 'scale(0.95)';
                    setTimeout(() => {
                      e.target.style.transform = '';
                    }, 150);
                  }}
                >
                  <span>🌐</span> Explore Github
                </a>

                <a
                  href="/docs/intro"
                  style={{
                    padding: '1rem 2.5rem',
                    borderRadius: '10px',
                    border: '2px solid #374151',
                    color: isLightMode ? '#1f2937' : 'white',
                    fontWeight: '600',
                    textDecoration: 'none',
                    transition: 'all 0.3s ease',
                    display: 'inline-flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '0.7rem',
                    minWidth: '180px'
                  }}
                  onMouseEnter={(e) => {
                    e.target.style.borderColor = '#E00070';
                    if (isLightMode) e.target.style.backgroundColor = 'rgba(224, 0, 112, 0.1)';
                    else e.target.style.backgroundColor = 'rgba(224, 0, 112, 0.1)';
                  }}
                  onMouseLeave={(e) => {
                    e.target.style.borderColor = '#374151';
                    e.target.style.backgroundColor = 'transparent';
                  }}
                  onClick={(e) => {
                    e.target.style.transform = 'scale(0.95)';
                    setTimeout(() => {
                      e.target.style.transform = '';
                    }, 150);
                  }}
                >
                  <span>📚</span> Read Book
                </a>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div style={{
          marginTop: '4rem',
          paddingTop: '2rem',
          borderTop: `1px solid ${isLightMode ? 'rgba(224, 0, 112, 0.2)' : 'rgba(224, 0, 112, 0.2)'}`
        }}>
          <div style={{
            maxWidth: '1400px',
            margin: '0 auto',
            padding: '0 1rem',
            textAlign: 'center'
          }}>
            <div style={{
              width: '80px',
              height: '2px',
              background: 'linear-gradient(45deg, #E00070, #FF7A00)',
              margin: '0.8rem auto',
              borderRadius: '2px'
            }}></div>
            <p style={{
              color: isLightMode ? '#9ca3af' : '#6b7280',
              fontSize: '0.8rem',
              marginBottom: '0.5rem'
            }}>
              Book Series & Hackathon • AI & Robotics Development
            </p>
            <p style={{
              color: isLightMode ? '#9ca3af' : '#6b7280',
              fontSize: '0.75rem'
            }}>
              © 2025 RoboMind AI Textbook
            </p>
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default function Home() {
  return <HomePage />;
}