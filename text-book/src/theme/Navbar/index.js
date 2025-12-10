import React from 'react';
import Navbar from '@theme-original/Navbar';

// Simple placeholder for auth controls - will be replaced later
const SafeAuthControls = () => {
  return (
    <div style={{
      display: 'flex',
      gap: '1rem',
      alignItems: 'center'
    }}>
      <a
        href="/signin"
        style={{
          color: 'white',
          textDecoration: 'none',
          padding: '0.5rem 1rem',
          border: '1px solid #E00070',
          borderRadius: '4px',
          fontSize: '0.9rem'
        }}
        onMouseEnter={(e) => {
          e.target.style.background = 'rgba(224, 0, 112, 0.1)';
        }}
        onMouseLeave={(e) => {
          e.target.style.background = 'transparent';
        }}
      >
        Sign In
      </a>
      <a
        href="/signup"
        style={{
          background: 'linear-gradient(45deg, #E00070, #FF7A00)',
          color: 'white',
          textDecoration: 'none',
          padding: '0.5rem 1rem',
          borderRadius: '4px',
          fontSize: '0.9rem'
        }}
      >
        Sign Up
      </a>
    </div>
  );
};

const CustomNavbar = (props) => {
  return (
    <>
      <Navbar {...props} />
      {/* Show account links in top right */}
      <div style={{
        position: 'fixed',
        top: '1rem',
        right: '1rem',
        zIndex: 1000
      }}>
        <SafeAuthControls />
      </div>
    </>
  );
};

export default CustomNavbar;