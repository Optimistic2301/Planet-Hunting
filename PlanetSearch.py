#Imports
import os ; os.environ['OMP_NUM_THREADS'] = os.environ.get('OMP_NUM_THREADS', '1')
import numpy as np
import os
import matplotlib.pyplot as plt
from pprint import pprint
from spirals import DiscModel
from frank.geometry import FixedGeometry
from scipy.signal import argrelextrema
from tqdm import tqdm
import contextlib
import sys
class DummyFile(object):
    def write(self, x): pass
@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout

#Functions
def uv_table_file_npz(filename, inc=67.4, PA=110.6, dRA=38e-3, dDec=-494e-3, point_source=2.4e-4, Rmax=2.0):   
    uvtable = filename
    geom = FixedGeometry(inc, PA, dRA, dDec)
    disc_model = DiscModel(
        uvtable, 
        Rmax=Rmax, 
        point_source=point_source, 
        geometry=geom, 
        fast_interp=True, 
        frank_params={'method': 'LogNormal', 'alpha': 1.2}
    )
    return disc_model
def find_planet(I, R, dist, plot=False):
    '''
    Finds local minima after peak to guess where planet might be orbiting.

    Args:
        - I: 1D array of intensity 
        - R: 1D array of radius
        - plot: Bool for displaying plots or just returning a float value in au.

    Returns:
        - radius (x) value of trough in au, and a plot displaying trough if plot=True
    '''
    try:
        if len(I) != len(R):
            raise ValueError("The length of intensity array (I) and radius array (R) must be equal.")
        peak_I = np.argmax(I)
        extrema = argrelextrema(I[peak_I:], np.less)
        if extrema[0].size == 0:
            raise IndexError("No minima found in the intensity data after the peak. Possible Index error")
        trough_I = extrema[0][0] + peak_I
        trough_R = R[trough_I]
        planet_au = trough_R * dist
        if plot:
            plt.plot(R, I, label= "Intensity emission [Jy/Sr]")
            plt.axvline(trough_R, color='g', linestyle='-', 
            label=f'''Possible planet at 
{trough_R:.2f} arcsec ({planet_au:.1f} AU)''')
            plt.xlabel('Radius [arcsec]')
            plt.ylabel('Intensity [Jy/Sr]')
            plt.title('Initial Planet search from intensity data')
            plt.legend()
            plt.show()
        r_planet = float(f"{trough_R:.2f}")
        r_ref = r_planet
        return r_planet, r_ref
    except IndexError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")              
def build_wakeflow(r_planet, r_ref, m_planet, hr):

    from wakeflow import WakeflowModel
    try:
        planet_model = WakeflowModel()
        planet_model.configure(
            name        = "lister_planet",
            system      = "DoAr25",
            m_star      = 1.0,
            m_planet    = m_planet,
            r_outer     = 1.5,
            r_inner     = 0.04,
            r_planet    = r_planet, # Estimated from I(R) plot
            r_ref       = r_ref, # Planet location seems sensible
            q           = 0.25,
            p           = 2.15,
            hr          = hr,    #heigh aspect ratio (bowtie) mainly governs the spiral production
            cw_rotation = False,
            grid_type   = "cylindrical",
            n_r         = 500,
            n_phi       = 2000,
            n_z         = 1,
            make_midplane_plots = False,
        )

    except Exception as e:
        
        raise RuntimeError(f'Could not configure WakeflowModel for {m_planet},{hr}. Error: {e}')
    
    with nostdout():
        planet_model.run(overwrite=True)
    
    return planet_model
def build_grid(model, clockwise = False, plot=False):
    DIR = f"{model.model_params['system']}/{model.model_params['name']}/{model.model_params['m_planet']}Mj/"
    R, PHI = np.load(DIR+'R.npy')[0,0,:], np.load(DIR+'PHI.npy')[:,0,0]
    drho = np.load(DIR + 'delta_rho.npy')[:, 0, :]

    if clockwise:
        drho = drho[::-1, :]

    rho_tot = np.load(DIR+'total_rho.npy')[:,0,:]
    drho = drho / (rho_tot - drho)
    
    if plot:   
        plt.pcolormesh(PHI, R, drho.T)
        plt.colorbar(label=r'$\Delta \rho / (\rho_{\text{total}} - \Delta \rho)$')
        plt.xlabel(r'$\phi$')
        plt.ylabel(r'$R$ (AU)')
        plt.title('Masked Grid')
        plt.show()
    
    return R, PHI, drho
def Intensity(model, R, drho, tau_p=1):
    kS = model.model_params['p'] + model.model_params['q'] - 1.5
    tau = tau_p * (R/model.model_params['r_planet'])**-kS
    dSigma = drho 
    dSigma = np.maximum(dSigma, 0) #Make sure that the perturbed Sigma >= 0.
    dtau =  dSigma.copy() 
    I0   = -np.expm1(-tau).reshape(1,-1) # (1-exp(-tau0))
    Itot = -np.expm1(-(1+dtau)*tau)      # (1-exp(-tau))
    dI = (Itot - I0) / I0
    return dI , Itot
def calculate_chi_squares(planet_model, disc_model, x, y, z, num_positions,
    mass, hr, save_dir, radius, clockwise = False, start=0, end=2*np.pi
    ):
    """
    Calculate and store X_2 results for a single combination of planet mass and aspect ratio.

    Args:
        planet_model: Planet model object.
        disc_model: Disc model object.
        x, y, z: Coordinates.
        num_positions: Number of positions to evaluate.
        mass: Planet mass.
        hr: Aspect ratio.
        save_dir: Directory to save results.
        start: Start angle (default: 0). rads
        end: End angle (default: 2π). rads

    Returns:
        Dictionary containing the X_2 results for this mass and hr combination.
    """
    
    planet_model.model_params['m_planet'] = mass
    planet_model.model_params['hr'] = hr
    planet_model.model_params['r_planet'] = radius

    positions = np.linspace(start, end, num_positions)
    
    model_chi_2_array = np.array([
        disc_model.make_perturbed_image(x, y, z.T, rotate=i)[1] for i in positions
    ])
    obs_chi_2_array = np.array([disc_model.chi2axi] * num_positions)
    X_2 = model_chi_2_array - obs_chi_2_array

    result = {
        "X_2": X_2,
        "obs_chi_2_array": obs_chi_2_array,
        "model_chi_2_array": model_chi_2_array
    }
    if clockwise == False:
        filename = f"{save_dir}/X2_radius_{radius:.3f}_mass_{mass:.3f}_hr_{hr:.3f}.npz"
        np.savez(filename, **result)
    else:
        filename = f"{save_dir}/X2_radius_{radius:.3f}_mass_{mass:.3f}_hr_{hr:.3f}_clockwise.npz" 
        np.savez(filename, **result)

    return result
def planet_hunting(
    filename,
    distance=145,
    masses=np.linspace(0.03, 0.5, 2),
    aspect_ratios=np.linspace(0.05, 0.1, 2),
    num_positions=20,
    radii_positions = 20,
    clockwise=False,
    plot=False,
    save_dir="/home/project/DSHARP + Spirals/Replacement/Output"
):
    """
    Planet-hunting workflow that evaluates multiple planet masses and aspect ratios,
    saving results into .npz files for recovery and further analysis.
    """
    os.makedirs(save_dir, exist_ok=True)

    disc_model = uv_table_file_npz(filename)
    print('Disc model built')
    initial_r_planet, r_ref = find_planet(disc_model.I_1D, disc_model.R, distance, plot)
    results_dict = {}
    
    r_inner, r_outer = 0.8 * initial_r_planet, 1.2 * initial_r_planet
    radii = np.linspace(r_inner, r_outer, radii_positions)  

    for r_planet in tqdm(radii, desc="Processing radii"):
        print(f"Processing radius {r_planet:.3f}")
        for mass in tqdm(masses, desc="Processing masses"):
            print(f"Clockwise = {clockwise}")
            for hr in tqdm(aspect_ratios, desc="Processing aspect ratios", leave=False):
                print('Building wakeflow...')
                planet_model = build_wakeflow(r_planet, r_ref, m_planet=mass, hr=hr)
                print('Configuring grid...')
                R, PHI, drho = build_grid(planet_model, clockwise, plot=plot)
                print('Calculating intensity...')
                dI, Itot = Intensity(planet_model, R, drho)
                print(f'Running chi analysis for mass: {mass} Mj and aspect ratio: {hr}...')
                try:
                    result = calculate_chi_squares(
                        planet_model, disc_model, R, PHI, dI, num_positions, 
                        mass, hr, save_dir, r_planet, clockwise
                    )
                except Exception as e:
                    print(f"Error encountered for mass={mass}, hr={hr}: Error = {e}")
                    continue

                results_dict[(r_planet, mass, hr)] = result 

                del R, PHI, drho, dI, Itot

    pprint(results_dict)
    print(f'Planet Hunting finished, files stored in {save_dir}')

    return results_dict
def dictionary_builder(filepath):
    data_dict = {}
    for f in os.listdir(filepath):
        if f.endswith('.npz'):
            parts = f.replace('.npz', '').split('_')
            
            if len(parts) < 5:
                print(f"Unexpected filename format: {f}")
                continue
            clockwise_flag = 'clockwise' in parts
            try:
                mass_value = float(parts[2])
                aspect_ratio_value = float(parts[4])
            except ValueError:
                print(f"File with invalid parameters: {f}")
                continue
            file_path = os.path.join(filepath, f)
            load_data = np.load(file_path)
            if mass_value not in data_dict:
                data_dict[mass_value] = {}
            if aspect_ratio_value not in data_dict[mass_value]:
                data_dict[mass_value][aspect_ratio_value] = {'clockwise': None, 'anticlockwise': None}
            if clockwise_flag:
                data_dict[mass_value][aspect_ratio_value]['clockwise'] = {key: load_data[key] for key in load_data}
            else:
                data_dict[mass_value][aspect_ratio_value]['anticlockwise'] = {key: load_data[key] for key in load_data}
    print("Data Dictionary:")
    for mass, hr_dict in data_dict.items():
        pprint(f"Mass: {mass}")
        for aspect_ratio, directions in hr_dict.items():
            for direction, data in directions.items():
                if data is not None:
                    pprint(f"  Aspect Ratio: {aspect_ratio}, {direction}, Keys: {list(data.keys())}")
    return data_dict
def dictionary_builder_radius(filepath):
    """
    Build a nested dictionary from .npz files with keys for radius, mass, and aspect ratio.
    """
    data_dict = {}
    for f in os.listdir(filepath):
        if f.endswith('.npz'): 
            parts = f.replace('.npz', '').split('_')
            if len(parts) < 7: 
                print(f"Unexpected filename format: {f}")
                continue
            try:
                radius_value = float(parts[2]) 
                mass_value = float(parts[4])   
                aspect_ratio_value = float(parts[6])
            except ValueError:
                print(f"File with invalid parameters: {f}")
                continue
            clockwise_flag = 'clockwise' in parts
            file_path = os.path.join(filepath, f)
            load_data = np.load(file_path)
            if radius_value not in data_dict:
                data_dict[radius_value] = {}
            if mass_value not in data_dict[radius_value]:
                data_dict[radius_value][mass_value] = {}
            data_dict[radius_value][mass_value][aspect_ratio_value]= {
                'clockwise': clockwise_flag,
                'data':{key: load_data[key] for key in load_data}
                }
    print("Data Dictionary:")
    for radius, mass_dict in data_dict.items():
        pprint(f"Radius: {radius}")
        for mass, hr_dict in mass_dict.items():
            pprint(f"  Mass: {mass}")
            for aspect_ratio, data in hr_dict.items():
                if data['clockwise']:
                    pprint(f"Aspect Ratio: {aspect_ratio}, Clockwise: True, Keys: {list(data.keys())}")
                else:
                    pprint(f"Aspect Ratio: {aspect_ratio}, Keys: {list(data.keys())}")
    return data_dict
def make_plots_from_dict(data_dict):

    anticlockwise_data = {}
    clockwise_data = {}
    
    for mass, ar_dict in data_dict.items():
        for aspect_ratio, data in ar_dict.items():
            if data['clockwise']:
                if mass not in clockwise_data:
                    clockwise_data[mass] = {}
                clockwise_data[mass][aspect_ratio] = data
            else:
                if mass not in anticlockwise_data:
                    anticlockwise_data[mass] = {}
                anticlockwise_data[mass][aspect_ratio] = data

    for mass, ar_dict in anticlockwise_data.items():
        plt.figure(figsize=(8, 5))
        for aspect_ratio, data in ar_dict.items():
            X_2_data = data['data']['X_2']
            if isinstance(X_2_data, np.ndarray):
                x = np.linspace(0, 2 * np.pi, len(X_2_data))
                plt.plot(x, X_2_data, label=f"Aspect Ratio: {aspect_ratio}")
            else:
                print(f"Skipping Mass: {mass}, Aspect Ratio: {aspect_ratio}, unexpected format for X_2.")
        plt.title(r"Anticlockwise $\chi^2$ for Planet mass of {}".format(mass))
        plt.xlabel(r'$\Phi$')
        plt.ylabel(r'$\chi^2$')
        plt.legend(loc='upper right', fontsize='small', ncol=2, frameon=False)
        plt.show()

    for mass, ar_dict in clockwise_data.items():
        plt.figure(figsize=(8, 5))
        for aspect_ratio, data in ar_dict.items():
            X_2_data = data['data']['X_2']
            if isinstance(X_2_data, np.ndarray):
                x = np.linspace(0, 2 * np.pi, len(X_2_data))
                plt.plot(x, X_2_data, label=f"Aspect Ratio: {aspect_ratio}")
            else:
                print(f"Skipping Mass: {mass}, Aspect Ratio: {aspect_ratio}, unexpected format for X_2.")
        plt.title(r"Clockwise $\chi^2$ for Planet mass of {}".format(mass))
        plt.xlabel(r'$\Phi$')
        plt.ylabel(r'$\chi^2$')
        plt.legend(loc='upper right', fontsize='small', ncol=2, frameon=False)
        plt.show()
def get_lowest_chi_and_corresponding_arrays(data_dict_full, plot=False):
    """
    Finds the lowest X_2 value for anticlockwise for each mass and aspect ratio,
    and retrieves the corresponding clockwise array for the same mass and aspect ratio.
    Returns a list of tuples:
    (mass, best_aspect_ratio, min_x2_acw, acw_array, corresponding_cw_array).
    """
    results = []
    
    for radius in sorted(data_dict_full.keys()):  
        for mass in sorted(data_dict_full[radius].keys()):  
            lowest_x2_acw = None
            best_aspect_ratio = None
            acw_array = None
            corresponding_cw_array = None
            for aspect_ratio, data in data_dict_full[radius][mass].items():
                try:
                    if not data.get('clockwise', False):  
                        current_x2_acw = data['data']['X_2']
                        min_x2_acw = np.min(current_x2_acw)
                        
                        if lowest_x2_acw is None or min_x2_acw < lowest_x2_acw:
                            lowest_x2_acw = min_x2_acw
                            best_aspect_ratio = aspect_ratio
                            acw_array = current_x2_acw
                            for cw_aspect, cw_data in data_dict_full[radius][mass].items():
                                if cw_data.get('clockwise', False): 
                                    corresponding_cw_array = cw_data['data']['X_2']
                                    break 

                except KeyError as e:
                    print(f"Missing key: {e} for radius {radius}, mass {mass}, aspect_ratio {aspect_ratio}")
                    continue
            
            if lowest_x2_acw is not None and corresponding_cw_array is not None:
                results.append((radius, mass, best_aspect_ratio, lowest_x2_acw, acw_array, corresponding_cw_array))
            else:
                results.append((radius, mass, best_aspect_ratio, lowest_x2_acw, acw_array, None)) 
    
    if plot:
        for radius, mass, aspect_ratio, min_x2_acw, acw_array, cw_array in results:
            if acw_array is not None and cw_array is not None:
                plt.figure(figsize=(8, 6))
                plt.plot(np.linspace(0, 2*np.pi, len(acw_array)),
                         acw_array, label=f"Anticlockwise, min: {np.min(acw_array):.2f}", color="blue")
                plt.plot(np.linspace(0, 2*np.pi, len(cw_array)),
                         cw_array, label=f"Clockwise min: {np.min(cw_array):.2f}", color="orange")
                plt.title(f"Radius {radius}, Mass {mass}, Aspect Ratio {aspect_ratio}")
                plt.xlabel(r"$\Phi$")
                plt.ylabel(r"$\chi^2$")
                plt.legend(frameon=False)
                plt.show()
    
    return results

def make_disc_heatmap(plot_array, threshold=1, clockwise = False):
    """
    Create a disc-like heatmap with polar projection for chi-square values.

    Args:
        plot_array: List of tuples in the form (radius, mass, aspect_ratio, chi_value, azimuthal_array).
        threshold: The chi-square threshold value where colors transition to blue.
    """
    if clockwise == False:
        plot_array = [item for item in plot_array if item[0] is not None and item[3] is not None and item[4] is not None]
        radii = sorted(set(item[0] for item in plot_array))  
        num_azimuthal_points = len(plot_array[0][5]) 
        azi_array = item[4] in plot_array
    else:
        plot_array = [item for item in plot_array if item[0] is not None and item[3] is not None and item[5] is not None]
        num_azimuthal_points = len(plot_array[0][4]) 
        radii = sorted(set(item[0] for item in plot_array))
        num_azimuthal_points = len(plot_array[0][5]) 
        azi_array = item[5] in plot_array
    theta = np.linspace(0, 2 * np.pi, num_azimuthal_points)
    Z = np.zeros((len(radii), num_azimuthal_points))

    for item in plot_array:
        radius, chi_value, azimuthal_array = item[0], item[3], azi_array
        radius_idx = radii.index(radius)  
        
        adjusted_chi_values = np.where(azimuthal_array > threshold, threshold, azimuthal_array)
        Z[radius_idx, :] = adjusted_chi_values

    R, Theta = np.meshgrid(radii, theta)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 6))
    heatmap = ax.pcolormesh(Theta, R, Z.T, cmap='RdBu', shading='auto', vmin=np.min(Z), vmax=threshold)
    ax.grid(axis='x')
    ax.set_theta_offset(np.pi)
    ax.set_theta_direction(-1)
    ax.set_rgrids([])
    ax.set_rticks([])
    ax.set_rgrids([0.10,0.30,0.50,0.70,0.90], angle=180)
    ax.set_xticklabels([])
    cbar = plt.colorbar(heatmap, ax=ax, pad=0.1)
    cbar.set_label(r"$\chi^2$ Value", fontsize=12)
    ax.set_title(r"Disc Heatmap of $\chi^2$ Values", va='bottom', fontsize=14)
    plt.show()

def make_disc_heatmap2(plot_array, threshold=1):
    """
    Create disc-like heatmaps with polar projection for chi-square values.

    Args:
        plot_array: List of tuples in the form (radius, mass, aspect_ratio, chi_value, azimuthal_array).
        threshold: The chi-square threshold value where colors transition to blue.
    """
    def create_heatmap(data, title, direction):
        radii = sorted(set(item[0] for item in data))
        num_azimuthal_points = len(data[0][direction])
        theta = np.linspace(0, 2 * np.pi, num_azimuthal_points)
        Z = np.zeros((len(radii), num_azimuthal_points))

        for item in data:
            radius, chi_value, azimuthal_array = item[0], item[3], item[direction]
            radius_idx = radii.index(radius)
            adjusted_chi_values = np.where(azimuthal_array > threshold, threshold, azimuthal_array)
            Z[radius_idx, :] = adjusted_chi_values

        R, Theta = np.meshgrid(radii, theta)

        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 6))
        heatmap = ax.pcolormesh(Theta, R, Z.T, cmap='RdBu', shading='auto', vmin=threshold-1000, vmax=threshold)

        ax.grid(axis='x')
        ax.set_theta_offset(np.pi)
        ax.set_theta_direction(-1 if direction == 4 else 1)
        ax.set_rticks([])
        ax.set_xlabel(r"Radius", fontsize=12)
        ax.set_rgrids([0.10, 0.30, 0.50, 0.70, 0.90], angle=180)
        ax.set_xticklabels([])
        
        cbar = plt.colorbar(heatmap, ax=ax, pad=0.1)
        cbar.set_label(r"$\chi^2$ Value", fontsize=12)
        ax.set_title(title, va='bottom', fontsize=14)

        plt.show()

    # Filter and validate plot_array
    acw_data = [item for item in plot_array if item[0] is not None and item[3] is not None and item[4] is not None]
    cw_data = [item for item in plot_array if item[0] is not None and item[3] is not None and item[5] is not None]

    if not acw_data:
        raise ValueError("No valid anticlockwise data in plot_array.")
    if not cw_data:
        raise ValueError("No valid clockwise data in plot_array.")

    create_heatmap(acw_data, r"Disc Heatmap of $\chi^2$ Values (Anticlockwise)", 4)
    create_heatmap(cw_data, r"Disc Heatmap of $\chi^2$ Values (Clockwise)", 5)

def calculate_bic(obs_chi2, model_chi2, num_data_points, num_params=4):
    """
    Calculate the BIC difference.
    
    Args:
        obs_chi2 (float): Total chi-squared value for the axisymmetric model.
        model_chi2 (float): 
        num_data_points (int): 
        num_params (int):

    Returns:
        float: Change in BIC (BIC_S - BIC_A)
    """
    if num_data_points <= 0:
        raise ValueError("Number of data points must be greater than zero.")

    delta_chi2 = obs_chi2 - model_chi2
    delta_bic = delta_chi2 - num_params * np.log(num_data_points)  
    return delta_bic


def make_disc_surface_plot(data_dict, threshold):
    """
    Create a disc-shaped 3D surface plot for X_2 values as a function of radius and azimuthal angle.

    Args:
        data_dict: The nested dictionary with structure {radius: {mass: {aspect_ratio: {}}}}.
        threshold: The maximum X_2 value to display on the plot.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    radii = sorted(data_dict.keys())
    max_azimuthal_points = 0

    for radius in radii:
        for mass, ar_dict in data_dict[radius].items():
            for aspect_ratio, data in ar_dict.items():
                X_2_data = data["X_2"]
                if isinstance(X_2_data, np.ndarray):
                    max_azimuthal_points = max(max_azimuthal_points, len(X_2_data))

    theta = np.linspace(0, 2 * np.pi, max_azimuthal_points)

    Z = np.full((len(radii), max_azimuthal_points), np.nan) 

    for i, radius in enumerate(radii):
        for mass, ar_dict in data_dict[radius].items():
            for aspect_ratio, data in ar_dict.items():
                X_2_data = data["X_2"]
                if not isinstance(X_2_data, np.ndarray):
                    continue

                masked_X_2 = np.where(X_2_data < threshold, X_2_data, np.nan)
                azimuthal_points = len(masked_X_2)
                Z[i, :azimuthal_points] = masked_X_2

    R, Theta = np.meshgrid(radii, theta)
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)   
    Z = -Z.T 

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='RdBu', edgecolor='none', rstride=1, cstride=1)
    cbar = fig.colorbar(surf, ax=ax, pad=0.1)
    cbar.set_label(r"$\Delta X_2$", fontsize=12)
    ax.set_title(r"Disc-Shaped 3D Surface Plot of $\Delta X_2$ Values", fontsize=14)

    ax.set_box_aspect([1, 1, 0.6]) 

    plt.show()
#Main
if __name__ == "__main__":
    print("Running Planet Hunting..")