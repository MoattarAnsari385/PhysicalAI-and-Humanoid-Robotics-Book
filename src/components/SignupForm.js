import React, { useState } from 'react';
import { useAuth } from '@better-auth/react';

const SignupForm = () => {
  const { signIn } = useAuth();
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    softwareLevel: '',
    gpuType: '',
    jetsonModel: ''
  });
  const [error, setError] = useState('');

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      // Create user with Better-Auth
      const result = await signIn('email-password', {
        email: formData.email,
        password: formData.password,
        user: {
          email: formData.email,
          name: formData.email.split('@')[0], // Use part of email as name
          softwareLevel: formData.softwareLevel,
          gpuType: formData.gpuType,
          jetsonModel: formData.jetsonModel
        }
      });

      if (result?.error) {
        setError(result.error.message);
      } else {
        // Redirect to Home/Library after successful signup
        window.location.href = '/home';
      }
    } catch (err) {
      setError(err.message || 'An error occurred during signup');
    }
  };

  return (
    <div style={{
      maxWidth: '500px',
      margin: '2rem auto',
      padding: '2rem',
      border: '1px solid #e0e0e0',
      borderRadius: '8px',
      fontFamily: 'system-ui, -apple-system, sans-serif'
    }}>
      <h2 style={{
        color: '#E00070',
        textAlign: 'center',
        marginBottom: '1.5rem'
      }}>
        Create Account
      </h2>

      {error && (
        <div style={{
          backgroundColor: '#fee',
          color: '#c33',
          padding: '0.75rem',
          borderRadius: '4px',
          marginBottom: '1rem'
        }}>
          {error}
        </div>
      )}

      <form onSubmit={handleSubmit}>
        <div style={{ marginBottom: '1rem' }}>
          <label htmlFor="email" style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500' }}>
            Email
          </label>
          <input
            type="email"
            id="email"
            name="email"
            value={formData.email}
            onChange={handleChange}
            required
            style={{
              width: '100%',
              padding: '0.75rem',
              border: '1px solid #ccc',
              borderRadius: '4px',
              fontSize: '1rem'
            }}
          />
        </div>

        <div style={{ marginBottom: '1rem' }}>
          <label htmlFor="password" style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500' }}>
            Password
          </label>
          <input
            type="password"
            id="password"
            name="password"
            value={formData.password}
            onChange={handleChange}
            required
            style={{
              width: '100%',
              padding: '0.75rem',
              border: '1px solid #ccc',
              borderRadius: '4px',
              fontSize: '1rem'
            }}
          />
        </div>

        <div style={{ marginBottom: '1rem' }}>
          <label htmlFor="softwareLevel" style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500' }}>
            Software Experience Level
          </label>
          <select
            id="softwareLevel"
            name="softwareLevel"
            value={formData.softwareLevel}
            onChange={handleChange}
            required
            style={{
              width: '100%',
              padding: '0.75rem',
              border: '1px solid #ccc',
              borderRadius: '4px',
              fontSize: '1rem'
            }}
          >
            <option value="">Select your level</option>
            <option value="beginner">Beginner</option>
            <option value="intermediate">Intermediate</option>
            <option value="advanced">Advanced</option>
          </select>
        </div>

        <div style={{ marginBottom: '1rem' }}>
          <label htmlFor="gpuType" style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500' }}>
            GPU Access (Optional)
          </label>
          <input
            type="text"
            id="gpuType"
            name="gpuType"
            value={formData.gpuType}
            onChange={handleChange}
            placeholder="e.g., RTX 4090, A100, etc."
            style={{
              width: '100%',
              padding: '0.75rem',
              border: '1px solid #ccc',
              borderRadius: '4px',
              fontSize: '1rem'
            }}
          />
        </div>

        <div style={{ marginBottom: '1.5rem' }}>
          <label htmlFor="jetsonModel" style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500' }}>
            Jetson Device (Optional)
          </label>
          <input
            type="text"
            id="jetsonModel"
            name="jetsonModel"
            value={formData.jetsonModel}
            onChange={handleChange}
            placeholder="e.g., Jetson Nano, Xavier NX, etc."
            style={{
              width: '100%',
              padding: '0.75rem',
              border: '1px solid #ccc',
              borderRadius: '4px',
              fontSize: '1rem'
            }}
          />
        </div>

        <button
          type="submit"
          style={{
            width: '100%',
            padding: '0.75rem',
            background: 'linear-gradient(45deg, #E00070, #FF7A00)',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            fontSize: '1rem',
            fontWeight: '500',
            cursor: 'pointer'
          }}
        >
          Sign Up
        </button>
      </form>

      <div style={{ textAlign: 'center', marginTop: '1rem' }}>
        <p>
          Already have an account?{' '}
          <a href="/login" style={{ color: '#E00070', textDecoration: 'none' }}>
            Sign in
          </a>
        </p>
      </div>
    </div>
  );
};

export default SignupForm;