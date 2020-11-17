/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <limits>

#include "helper_functions.h"

using std::string;
using std::vector;

static std::default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

  // Set the number of particles
  num_particles = 50;

  Particle temp_particle;
  
  // Create normal distribuition functions for the sensor noise
  std::normal_distribution<double> NormalDist_x(0, std[0]);
  std::normal_distribution<double> NormalDist_y(0, std[1]);
  std::normal_distribution<double> NormalDist_theta(0, std[2]);

  // Initialize each particle
  for (unsigned int i = 0; i < num_particles; ++i) {
    temp_particle.id = i;
    temp_particle.x = x + NormalDist_x(gen);
    temp_particle.y = y + NormalDist_y(gen);
    temp_particle.theta = theta + NormalDist_theta(gen);
    temp_particle.weight = 1.0;

    particles.push_back(temp_particle);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {

  // Create normal distribuition functions for the noise
  std::normal_distribution<double> NormalDist_x(0, std_pos[0]);
  std::normal_distribution<double> NormalDist_y(0, std_pos[1]);
  std::normal_distribution<double> NormalDist_theta(0, std_pos[2]);

  // Update the state of each particle using motion equation
  for (unsigned int i = 0; i < num_particles; ++i) {
    if (fabs(yaw_rate) < 0.00001) {  
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    } 
    else {
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }

    particles[i].x += NormalDist_x(gen);
    particles[i].y += NormalDist_y(gen);
    particles[i].theta += NormalDist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {

  for (unsigned int i = 0; i < observations.size(); ++i) {
    
    // Get the current observation
    LandmarkObs observation = observations[i];

    // Initialize the minimum distance to maximum possible
    double min_distance = std::numeric_limits<double>::max();

    int map_id = -1;
    
    for (unsigned int j = 0; j < predicted.size(); ++j) {
      // Get the current prediction
      LandmarkObs pred = predicted[j];
      
      // Calculate the distance between the current and predicted landmark
      double current_dist = dist(observation.x, observation.y, pred.x, pred.y);

      // Find the predicted landmark nearest the current observed landmark
      if (current_dist < min_distance) {
        min_distance = current_dist;
        map_id = pred.id;
      }
    }

    // Set the observation's id to the nearest predicted landmark's id
    observations[i].id = map_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {

  for (unsigned int i = 0; i < num_particles; ++i) {

    // Get the particle x, y coordinates
    double ptcl_x = particles[i].x;
    double ptcl_y = particles[i].y;
    double ptcl_theta = particles[i].theta;

    vector<LandmarkObs> predictions;

    for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); ++j) {
      float landmark_x = map_landmarks.landmark_list[j].x_f;
      float landmark_y = map_landmarks.landmark_list[j].y_f;
      int landmark_id = map_landmarks.landmark_list[j].id_i;
      
      // Get only landmarks within sensor range of the particle
      if (fabs(dist(landmark_x,landmark_y,ptcl_x,ptcl_y)) <= sensor_range) {
        predictions.push_back(LandmarkObs{ landmark_id, landmark_x, landmark_y });
      }
    }

    // List of observations transformed from vehicle coordinates to map coordinates
    vector<LandmarkObs> transformed_os;
    for (unsigned int j = 0; j < observations.size(); ++j) {
      double trans_x = cos(ptcl_theta)*observations[j].x - sin(ptcl_theta)*observations[j].y + ptcl_x;
      double trans_y = sin(ptcl_theta)*observations[j].x + cos(ptcl_theta)*observations[j].y + ptcl_y;
      transformed_os.push_back(LandmarkObs{ observations[j].id, trans_x, trans_y });
    }

    // Association for the predictions and transformed observations on current particle
    dataAssociation(predictions, transformed_os);

    // Initialize weight again
    particles[i].weight = 1.0;

    for (unsigned int j = 0; j < transformed_os.size(); j++) {      
      double obs_x, obs_y, pred_x, pred_y;
      obs_x = transformed_os[j].x;
      obs_y = transformed_os[j].y;

      int associated_prediction = transformed_os[j].id;

      // Get the x,y coordinates of the prediction associated with the current observation
      for (unsigned int k = 0; k < predictions.size(); ++k) {
        if (predictions[k].id == associated_prediction) {
          pred_x = predictions[k].x;
          pred_y = predictions[k].y;
        }
      }

      // Calculate weight for this observation with multivariate Gaussian
      double std_x = std_landmark[0];
      double std_y = std_landmark[1];
      double obs_w = multiv_prob(std_x, std_y, obs_x, obs_y, pred_x, pred_y);
      particles[i].weight *= obs_w;
    }
  }
}

void ParticleFilter::resample() {

 vector<Particle> new_particles;

  // Get all of the current weights
  vector<double> weights;
  for (unsigned int i = 0; i < num_particles; ++i) {
    weights.push_back(particles[i].weight);
  }

  std::uniform_int_distribution<int> uniintdist(0, num_particles-1);
  auto index = uniintdist(gen);

  // Get max weight
  double max_weight = *max_element(weights.begin(), weights.end());

  std::uniform_real_distribution<double> unirealdist(0.0, max_weight);

  double beta = 0.0;

  for (unsigned int i = 0; i < num_particles; ++i) {
    beta += unirealdist(gen) * 2.0;
    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    new_particles.push_back(particles[index]);
  }

  particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}