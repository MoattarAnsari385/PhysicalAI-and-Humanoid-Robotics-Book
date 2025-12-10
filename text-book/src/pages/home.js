import React from 'react';
import Layout from '@theme/Layout';
import ChapterControls from '../components/ChapterControls';

const HomePage = () => {
  return (
    <Layout title="Home" description="Your personalized robotics textbook">
      <div style={{
        minHeight: '100vh',
        fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
        background: 'linear-gradient(135deg, #111827 0%, #000000 50%, #111827 100%)',
        color: '#ffffff',
        padding: '2rem 0'
      }}>
        <div style={{
          maxWidth: '1200px',
          margin: '0 auto',
          padding: '0 2rem'
        }}>
          <header style={{
            textAlign: 'center',
            marginBottom: '3rem',
            padding: '2rem 0'
          }}>
            <h1 style={{
              fontSize: '2.5rem',
              fontWeight: '800',
              margin: '0 0 1rem 0',
              background: 'linear-gradient(45deg, #E00070, #FF7A00)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              backgroundClip: 'text',
              color: 'transparent'
            }}>
              Physical AI & Humanoid Robotics Textbook
            </h1>
            <p style={{
              fontSize: '1.2rem',
              color: '#9ca3af',
              maxWidth: '600px',
              margin: '0 auto'
            }}>
              Customize your learning experience with personalization and language options
            </p>
          </header>

          <div style={{
            marginBottom: '2rem',
            padding: '1.5rem',
            background: 'rgba(30, 41, 59, 0.3)',
            borderRadius: '12px',
            border: '1px solid rgba(224, 0, 112, 0.2)'
          }}>
            <ChapterControls />
          </div>

          <div style={{
            marginTop: '3rem',
            textAlign: 'center',
            padding: '2rem',
            borderTop: '1px solid rgba(224, 0, 112, 0.2)'
          }}>
            <h3 style={{ color: '#E00070', marginBottom: '1rem' }}>Continue Learning</h3>
            <p style={{ color: '#9ca3af', marginBottom: '1.5rem' }}>
              Explore the modules below to advance your robotics knowledge
            </p>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1.5rem' }}>
              <a
                href="/docs/intro"
                style={{
                  display: 'block',
                  padding: '1.5rem',
                  background: 'rgba(30, 41, 59, 0.5)',
                  borderRadius: '8px',
                  textDecoration: 'none',
                  color: '#d1d5db',
                  transition: 'transform 0.2s ease, box-shadow 0.2s ease',
                  border: '1px solid rgba(224, 0, 112, 0.2)'
                }}
                onMouseEnter={(e) => {
                  e.target.style.transform = 'translateY(-2px)';
                  e.target.style.boxShadow = '0 10px 25px rgba(224, 0, 112, 0.1)';
                }}
                onMouseLeave={(e) => {
                  e.target.style.transform = 'translateY(0)';
                  e.target.style.boxShadow = 'none';
                }}
              >
                <h4 style={{ margin: '0 0 0.5rem 0', color: '#E00070' }}>Getting Started</h4>
                <p style={{ margin: 0, fontSize: '0.9rem' }}>Introduction to the textbook and learning path</p>
              </a>
              <a
                href="/docs/module-1-ros2/intro"
                style={{
                  display: 'block',
                  padding: '1.5rem',
                  background: 'rgba(30, 41, 59, 0.5)',
                  borderRadius: '8px',
                  textDecoration: 'none',
                  color: '#d1d5db',
                  transition: 'transform 0.2s ease, box-shadow 0.2s ease',
                  border: '1px solid rgba(224, 0, 112, 0.2)'
                }}
                onMouseEnter={(e) => {
                  e.target.style.transform = 'translateY(-2px)';
                  e.target.style.boxShadow = '0 10px 25px rgba(224, 0, 112, 0.1)';
                }}
                onMouseLeave={(e) => {
                  e.target.style.transform = 'translateY(0)';
                  e.target.style.boxShadow = 'none';
                }}
              >
                <h4 style={{ margin: '0 0 0.5rem 0', color: '#E00070' }}>Module 1: ROS 2</h4>
                <p style={{ margin: 0, fontSize: '0.9rem' }}>Robotic Nervous System fundamentals</p>
              </a>
              <a
                href="/docs/module-2-simulation/intro"
                style={{
                  display: 'block',
                  padding: '1.5rem',
                  background: 'rgba(30, 41, 59, 0.5)',
                  borderRadius: '8px',
                  textDecoration: 'none',
                  color: '#d1d5db',
                  transition: 'transform 0.2s ease, box-shadow 0.2s ease',
                  border: '1px solid rgba(224, 0, 112, 0.2)'
                }}
                onMouseEnter={(e) => {
                  e.target.style.transform = 'translateY(-2px)';
                  e.target.style.boxShadow = '0 10px 25px rgba(224, 0, 112, 0.1)';
                }}
                onMouseLeave={(e) => {
                  e.target.style.transform = 'translateY(0)';
                  e.target.style.boxShadow = 'none';
                }}
              >
                <h4 style={{ margin: '0 0 0.5rem 0', color: '#E00070' }}>Module 2: Simulation</h4>
                <p style={{ margin: 0, fontSize: '0.9rem' }}>Digital Twin and Gazebo environments</p>
              </a>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default HomePage;