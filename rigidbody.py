import numpy as np
from scipy.spatial.transform import Rotation

from sound import *

class RigidBody:
    def __init__(self, model, CoM, pos, vel, accel, orient, ang_vel, ang_accel, mass, inertia):
        self.model = model
        for idx_v, v in enumerate(self.model.vertices):
            self.model.vertices[idx_v] = v - CoM
        self.CoM = CoM
        self.pos = pos
        self.vel = vel
        self.accel = accel
        self.orient = orient
        self.ang_vel = ang_vel
        self.ang_accel = ang_accel
        self.mass = mass
        self.inertia = inertia

    # shifts center of mass
    def shift_CoM(self, shift):
        for idx_v, v in enumerate(self.model.vertices):
            self.model.vertices[idx_v] = v - shift

        self.CoM = self.CoM - shift

    def update_mass(self, mdot, dt):
        self.mass += mdot * dt

    def apply_torque(self, torque):
        inertia_inverse = np.linalg.inv(self.inertia)
        accel = np.dot(inertia_inverse, torque)
        self.ang_accel = self.ang_accel + accel

    def apply_force(self, force):
        accel = force / self.mass
        self.accel = self.accel + accel

    def apply_accel(self, accel):
        self.accel = self.accel + accel

    def rotate(self, dt):
        if np.linalg.norm(self.ang_vel) > 0:
            # Ensure the angular velocity is a column vector
            # angular_velocity = self.ang_vel.reshape(3, 1)
            axis = self.ang_vel / np.linalg.norm(self.ang_vel)
            angle_rad = np.linalg.norm(self.ang_vel) * dt

            rotation = Rotation.from_rotvec(angle_rad * axis)
    
            # Convert the rotation to a rotation matrix
            rotation_matrix = rotation.as_matrix()
    
            # Multiply the original orientation matrix by the rotation matrix
            self.orient = np.dot(rotation_matrix, self.orient)

            self.orient[0] = self.orient[0] / np.linalg.norm(self.orient[0])
            self.orient[1] = self.orient[1] / np.linalg.norm(self.orient[1])
            self.orient[2] = self.orient[2] / np.linalg.norm(self.orient[2])

    def clear_accels(self):
        self.accel = np.array([0, 0, 0])
        self.ang_accel = np.array([0, 0, 0])

    def update(self, dt):
        self.vel = self.vel + self.accel * dt
        self.pos = self.pos + self.vel * dt
        self.ang_vel = self.ang_vel + self.ang_accel * dt
        self.rotate(dt)
        self.clear_accels()

class Rocket(RigidBody):
    def __init__(self, model, CoM, pos, vel, accel, orient, ang_vel, ang_accel, mass, inertia,
                 max_thrust, throttle_range, throttle, prop_mass, mass_flow, Cds, Cdas, cross_sections):
        super(Rocket, self).__init__(model, CoM, pos, vel, accel, orient, ang_vel, ang_accel, mass, inertia)
        self.max_thrust = max_thrust
        self.throttle_range = throttle_range
        self.throttle = throttle
        self.prop_mass = prop_mass
        self.mass_flow = mass_flow
        self.Cds = Cds
        self.Cdas = Cdas
        self.cross_sections = cross_sections

        self.aero_resistance = np.multiply(self.cross_sections, self.Cds)
        self.angular_resistance = np.multiply(self.cross_sections, self.Cdas)

        self.thrust = self.throttle / 100 * self.max_thrust

    def drain_fuel(self, dt):
        self.update_mass(-self.mass_flow * self.throttle / 100, dt)

    def apply_thrust(self):
        self.apply_force(self.orient[2] * self.thrust)

    def apply_drag(self):
        if np.linalg.norm(self.vel) > 0:
            drag_vector = -self.vel / np.linalg.norm(self.vel)
            drag_multiplier = abs(np.dot(drag_vector, self.orient[0] * self.aero_resistance[0])) + abs(np.dot(drag_vector, self.orient[1] * self.aero_resistance[1])) + abs(np.dot(drag_vector, self.orient[2] * self.aero_resistance[2]))
            drag_amount = 0.5 * drag_multiplier * np.linalg.norm(self.vel)**2
            force_vec = drag_vector * drag_amount
            self.apply_force(force_vec)

    def apply_aero_torque(self):
        vel_mag = np.linalg.norm(self.vel)
        if vel_mag:
            torque_x = np.dot(self.orient[1], self.vel) * vel_mag * self.Cdas[0]
            torque_y = -np.dot(self.orient[0], self.vel) * vel_mag * self.Cdas[1]

            self.apply_torque(np.array([torque_x, torque_y, 0]))

    def set_thrust_percent(self, percentage):
        if not percentage == 0:
            percentage = max(min(self.throttle_range[1], percentage), self.throttle_range[0])
        self.throttle = percentage
        self.thrust = self.max_thrust * percentage / 100

class Missile(RigidBody):
    def __init__(self, model, CoM, pos, vel, accel, orient, ang_vel, ang_accel, mass, inertia,
                 max_thrust, throttle_range, throttle, prop_mass, mass_flow, Cds, Cdas, cross_sections, target=None):
        super(Missile, self).__init__(model, CoM, pos, vel, accel, orient, ang_vel, ang_accel, mass, inertia)
        self.max_thrust = max_thrust
        self.throttle_range = throttle_range
        self.throttle = throttle
        self.prop_mass = prop_mass
        self.mass_flow = mass_flow
        self.Cds = Cds
        self.Cdas = Cdas
        self.cross_sections = cross_sections
        self.target = target

        self.dry_mass = self.mass - self.prop_mass

        self.aero_resistance = np.multiply(self.cross_sections, self.Cds)
        self.angular_resistance = np.multiply(self.cross_sections, self.Cdas)

        self.thrust = self.throttle / 100 * self.max_thrust
        self.hp = 1

        self.trail = []
        self.boost_trail = []
        self.payload_offset = 5

    def update_trail(self):
        if self.prop_mass > 0 and self.thrust > 0:
            self.boost_trail.append(self.pos)
        else:
            self.trail.append(self.pos)

    def check_target(self, bodies):
        if self.target and np.linalg.norm(self.pos - self.target.pos) < 15:
            self.target.hp -= 50
            bodies.remove(self.target)
            del self.target
            bodies.remove(self)
            play_sfx("strike", channel=random.randint(4,6))
            del self

    def guidance(self, dt):
        if (not self.target) or np.linalg.norm(self.vel) == 0:
            return

        t = self.target
        m = self

        if np.dot(t.pos - m.pos, self.orient[2]) < 0.1:
            return

        K_p = 1
        K_v = 0.001
        K_a = 1

        aimpoint = 1 * t.pos + t.vel * K_v * np.linalg.norm(t.pos - m.pos) + K_a * t.accel * np.linalg.norm(t.pos - m.pos)
        aimpoint_dir = aimpoint - m.pos
        aimpoint_dir /= np.linalg.norm(aimpoint_dir)

        m.vel = m.vel * (1 - dt * 2) + (aimpoint_dir * np.linalg.norm(m.vel)) * dt * 2
        
    def drain_fuel(self, dt):
        if self.prop_mass > 0:
            self.update_mass(-self.mass_flow * self.throttle / 100, dt)
            self.prop_mass -= self.mass_flow * self.throttle / 100 * dt

    def apply_thrust(self):
        if self.prop_mass > 0:
            self.apply_force(self.orient[2] * self.thrust)

    def apply_drag(self):
        if np.linalg.norm(self.vel) > 0:
            drag_vector = -self.vel / np.linalg.norm(self.vel)
            drag_multiplier = abs(np.dot(drag_vector, self.orient[0] * self.aero_resistance[0])) + abs(np.dot(drag_vector, self.orient[1] * self.aero_resistance[1])) + abs(np.dot(drag_vector, self.orient[2] * self.aero_resistance[2]))
            drag_amount = 0.5 * drag_multiplier * np.linalg.norm(self.vel)**2
            force_vec = drag_vector * drag_amount
            self.apply_force(force_vec)

    def apply_aero_torque(self):
        vel_mag = np.linalg.norm(self.vel)
        if vel_mag:
            torque_x = np.dot(self.orient[1], self.vel) * vel_mag * self.Cdas[0]
            torque_y = -np.dot(self.orient[0], self.vel) * vel_mag * self.Cdas[1]

            self.apply_torque(np.array([torque_x, torque_y, 0]))

    def set_thrust_percent(self, percentage):
        if not percentage == 0:
            percentage = max(min(self.throttle_range[1], percentage), self.throttle_range[0])
        self.throttle = percentage
        self.thrust = self.max_thrust * percentage / 100

class SimpleAircraft(RigidBody):
    def __init__(self, model, CoM, pos, vel, accel, orient, ang_vel, ang_accel, mass, inertia,
                 max_thrust, throttle_range, throttle, prop_mass, mass_flow,
                 cross_sections, Cds, Cdas, angular_damping, Cl, lift_moment_arm,
                 control_effectiveness, rear_gear_moment, brake_force, weapons):
        super(SimpleAircraft, self).__init__(model, CoM, pos, vel, accel, orient, ang_vel, ang_accel, mass, inertia)
        self.max_thrust = max_thrust
        self.throttle_range = throttle_range
        self.throttle = throttle
        self.prop_mass = prop_mass
        self.mass_flow = mass_flow
        self.cross_sections = cross_sections
        self.Cds = Cds
        self.Cdas = Cdas
        self.angular_damping = angular_damping # this is for the complicated aero effects which I can not simulate using the single body model
        self.Cl = Cl
        self.lift_moment_arm = lift_moment_arm
        self.control_effectiveness = control_effectiveness
        self.rear_gear_moment = rear_gear_moment
        self.brake_force = brake_force

        self.weapons = weapons

        self.aero_resistance = np.multiply(self.cross_sections, self.Cds)
        self.angular_resistance = np.multiply(self.cross_sections, self.Cdas)

        self.thrust = self.throttle / 100 * self.max_thrust

        self.brake = 0

    def drain_fuel(self, dt):
        self.update_mass(-self.mass_flow * self.throttle / 100, dt)
        self.prop_mass = self.prop_mass - self.mass_flow * self.throttle / 100 * dt
        if self.prop_mass <= 0:
            self.prop_mass = 0
            self.thrust = 0

    def apply_thrust(self):
        self.apply_force(self.orient[2] * self.thrust)

    def set_thrust_percent(self, percentage):
        if not percentage == 0:
            percentage = max(min(self.throttle_range[1], percentage), self.throttle_range[0])
        self.throttle = percentage
        self.thrust = self.max_thrust * percentage / 100

    def apply_aero_torque(self):
        vel_mag = np.linalg.norm(self.vel)
        if vel_mag:
            torque_x = np.dot(self.orient[1], self.vel) * vel_mag * self.Cdas[0]
            torque_y = -np.dot(self.orient[0], self.vel) * vel_mag * self.Cdas[1]

            self.apply_torque(np.array([torque_x, torque_y, 0]))

    def apply_angular_drag(self, dt):
        drag_vector = self.ang_vel * np.linalg.norm(self.ang_vel)
        drag_multiplier = abs(np.dot(drag_vector, self.orient[0] * self.angular_resistance[1])) + abs(np.dot(drag_vector, self.orient[1] * self.angular_resistance[0])) + abs(np.dot(drag_vector, self.orient[2] * self.angular_resistance[2]))
        drag_vector = -drag_vector * drag_multiplier**2
        self.apply_torque(drag_vector)

        # this is placeholder for the complicated aero effects which I can not simulate using the single body model
        self.ang_vel[0] = self.ang_vel[0] * (1 - self.angular_damping[0] * dt)
        self.ang_vel[1] = self.ang_vel[1] * (1 - self.angular_damping[1] * dt)
        self.ang_vel[2] = self.ang_vel[2] * (1 - self.angular_damping[2] * dt)

    def apply_lift(self):
        if np.linalg.norm(self.vel) > 0:
            vel_mag = np.linalg.norm(self.vel)

            yz_vel = self.vel - self.orient[0] * np.dot(self.vel, self.orient[0])
            yz_mag = np.linalg.norm(yz_vel)
            
            AoA = np.arccos(max(min(np.dot(yz_vel, self.orient[2]) / yz_mag, 1), -1))
            AoA = np.rad2deg(AoA)

            if abs(AoA) < 20:
                lift_multiplier = abs(AoA) / 20
                if np.dot(self.orient[1], self.vel) > 0:
                    lift_multiplier = lift_multiplier * -1
            elif abs(AoA) < 40:
                lift_multiplier = (AoA - 20) / 20
                if np.dot(self.orient[1], self.vel) > 0:
                    lift_multiplier = lift_multiplier * -1
            else:
                return

            force_vec = self.orient[1] - (self.vel / vel_mag) * np.dot(self.orient[1], self.vel / vel_mag)
            if np.linalg.norm(force_vec) > 0:
                force_vec = force_vec / np.linalg.norm(force_vec)
                force = lift_multiplier * self.Cl * 0.5 * self.cross_sections[1] * vel_mag**2

                force_vec = force_vec * force

                self.apply_force(force_vec)
                self.apply_torque(np.array([force * self.lift_moment_arm, 0, 0]))
                
    def apply_drag(self):
        if np.linalg.norm(self.vel) > 0:
            drag_vector = -self.vel / np.linalg.norm(self.vel)
            drag_multiplier = abs(np.dot(drag_vector, self.orient[0] * self.aero_resistance[0])) + abs(np.dot(drag_vector, self.orient[1] * self.aero_resistance[1])) + abs(np.dot(drag_vector, self.orient[2] * self.aero_resistance[2]))
            drag_amount = 0.5 * drag_multiplier * np.linalg.norm(self.vel)**2
            force_vec = drag_vector * drag_amount
            self.apply_force(force_vec)

    def aileron(self, direction):
        self.apply_torque(np.array([0, 0, 1]) * direction * np.linalg.norm(self.vel)**2 * self.control_effectiveness[0])

    def elevator(self, direction):
        self.apply_torque(np.array([1, 0, 0]) * direction * np.linalg.norm(self.vel)**2 * self.control_effectiveness[1])

    def rudder(self, direction):
        self.apply_torque(np.array([0, 1, 0]) * direction * np.linalg.norm(self.vel)**2 * self.control_effectiveness[2])

    def update_throttle(self, direction, dt):
        self.throttle = self.throttle + direction * dt
        self.throttle = max(min(self.throttle_range[1], self.throttle), self.throttle_range[0])
        self.thrust = self.throttle / 100 * self.max_thrust

    def update_weapons(self, dt):
        for w in self.weapons:
            w.update(dt)

class KKV(RigidBody):
    def __init__(self, model, CoM, pos, vel, accel, orient, ang_vel, ang_accel, mass, inertia,
                 thrust, max_thrust, ACS_moment, max_ACS_moment):
        super(KKV, self).__init__(model, CoM, pos, vel, accel, orient, ang_vel, ang_accel, mass, inertia)
        self.thrust = thrust
        self.max_thrust = max_thrust
        self.ACS_moment = ACS_moment
        self.max_ACS_moment = max_ACS_moment

        self.target = None
        self.locdir = None
        self.active = False
        self.carrier = None

    def divert(self, locdir):
        if self.active:
            self.thrust = np.array([0, 0, 0])
            
            if "right" in locdir:
                self.thrust = self.thrust + self.orient[0] * self.max_thrust

            if "left" in locdir:
                self.thrust = self.thrust - self.orient[0] * self.max_thrust

            if "up" in locdir:
                self.thrust = self.thrust + self.orient[1] * self.max_thrust

            if "down" in locdir:
                self.thrust = self.thrust - self.orient[1] * self.max_thrust

            self.apply_force(self.thrust)
            self.locdir = locdir
            
        else:
            self.locdir = None

    def control_attitude(self):
        if self.target:
            target_rel_pos = self.target.pos - self.pos
            target_dist = np.linalg.norm(target_rel_pos)
            direction = target_rel_pos / target_dist
            self.orient[2] = direction
            # orient2_plane = self.orient[2] - np.array([0, 1, 0]) * np.dot(self.orient[2], np.array([0, 1, 0]))
            # self.orient[0] =

            v = self.orient[2]

            if np.allclose(self.orient[2], [1, 0, 0]):
                temp_vector = np.array([0, 1, 0])
            else:
                temp_vector = np.array([1, 0, 0])

            # Step 2: Calculate u1 as the cross product of v and the temp_vector
            u1 = np.cross(v, temp_vector)

            # Step 3: Calculate u2 as the cross product of v and u1
            u2 = np.cross(v, u1)

            # Step 4: Ensure the y-component of u2 is zero
            u2[1] = 0  # setting y-component to 0

            # Optional: Normalize u1 and u2
            u1 = u1 / np.linalg.norm(u1)
            u2 = u2 / np.linalg.norm(u2)

            self.orient[0] = u1
            self.orient[1] = u2

